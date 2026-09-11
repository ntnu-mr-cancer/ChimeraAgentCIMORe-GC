# Architecture

## Graph

```
HumanMessage(prompt)
        │
        ▼
  ┌──────────┐  tool_calls    ┌──────────┐
  │  agent   │───────────────▶│  tools   │  (MCP, stdio)
  │  (LLM)   │◀───────────────│          │
  └────┬─────┘   results      └──────────┘
       │ no tool_calls
       ▼
  ┌──────────────┐
  │ form_fill    │  same model + per-case Pydantic schema
  │ (LLM)        │  validated via PydanticOutputParser; retries on
  │              │  validation error (agent.form_fill.max_retries)
  └──────┬───────┘
         ▼
       END  →  structured_response (the validated per-case record)
```

The same model is used for both ReAct turns and the terminal form-fill;
the form-fill node prompts it with a concrete JSON skeleton (not a JSON
schema dump — small models echo schemas back). This is provider-neutral:
any LangChain `BaseChatModel` works.

**Source**: `src/chimera_agent_baseline/agent/graph.py`,
`src/chimera_agent_baseline/run.py`.

## MCP tool server

All tools are exposed via the [Model Context Protocol](https://modelcontextprotocol.io).
The server runs as a stdio subprocess and is framework-agnostic.

**Source**: `src/chimera_agent_baseline/mcp_server.py`.

One registry per task, selected automatically from the task number the
runner is processing (`task1` / `task2` / `task3`):

Tasks 1 and 2 expose the same six tools — the masked "Extended EHR view"
documents the urologist could reveal — plus guideline search. Task 3
drops the PSA trend and lab panel and adds the surgical (prostatectomy)
pathology report:

| Tool | Task 1 | Task 2 | Task 3 | Returns |
|---|---|---|---|---|
| `get_psa_trend` | ✓ | ✓ | — | Prior PSA values as a time series |
| `get_lab_results` | ✓ | ✓ | — | Full lab panel |
| `get_mri_report` | ✓ | ✓ | ✓ | mpMRI report prose |
| `get_pathology_report` | ✓ | ✓ | ✓ | Biopsy report prose (task 1: "no data" when no prior biopsy) |
| `get_surgical_pathology_report` | — | — | ✓ | Radical-prostatectomy pathology prose |
| `get_previous_notes` | ✓ | ✓ | ✓ | Prior GP / urology notes |
| `get_family_history` | ✓ | ✓ | ✓ | First-degree PCa history |
| `search_guidelines` | ✓ | ✓ | ✓ | Semantic search over EAU corpus |

The structured headline values (PI-RADS, PSA density, volume, csPCa,
Gleason / ISUP, …) are in `prompt.json` up front, so the tools serve only
the free-text documents behind them.

### Adding a custom tool

Define a `ToolSpec` in `src/chimera_agent_baseline/tools/definitions.py`
and append it to `TASK1_TOOLS` (or `TASK2_TOOLS`):

```python
MY_TOOL = ToolSpec(
    name="get_my_data",
    description="Retrieve my custom data for a patient case.",
    fields=("my_field", "another_field"),
)
```

`fields` lists the top-level keys from each case's `clinical.json` that
this tool should return. Missing keys are silently omitted (so a
biopsy-naïve case calling `get_pathology_report` returns just
`{case_id}` plus a "no data" note). For tools that don't follow the
precomputed-data pattern (e.g. an API call), register them directly in
`mcp_server.py` with the `@mcp.tool()` decorator (see
`search_guidelines`).

### What you can change

You're free to **add** tools, change a tool's `description`, or
expand its `fields`. Be aware that `output/schema.py` maps each
tool-gated reasoning variable to the tool that backs it (e.g.
`family_history → get_family_history`): renaming or removing that tool
makes the variable un-rateable for any case. The headline values
(PI-RADS, PSA density, Gleason / ISUP, …) are in `prompt.json`, so those
variables are always rateable regardless of tools.

## Tools are optional

Only the final structured output is evaluated — tool calls are not scored.
But the MCP tools reveal the masked EHR documents behind each case, so
calling them gives the agent more evidence to reason over. Use them
because they improve the decision, not because they're graded.

## Feature embeddings

Each patient may ship a single `features.json` alongside `prompt.json` /
`clinical.json`, holding frozen foundation-model image embeddings,
separated by origin (JSON attribute). Each origin is a **list of feature
vectors** (a list of JSON arrays), uniform across origins:

```jsonc
{
  "mri":           [[...]],          // one vector,  all tasks
  "biopsy":        [[...], [...]],   // one or more, tasks 2 & 3
  "prostatectomy": [[...], [...]]    // one or more, task 3 only
}
```

Vectors are raw foundation-model output (e.g. 960-d) and should **not**
enter the LLM context directly — build a predictor or tool on top and feed
the agent a compact score/label. The baseline does not consume features.

`chimera_agent_baseline.features.FeatureStore` is a decoupled loader (no
ties to the agent graph or MCP server): it indexes `features.json` by
`case_id` and exposes `get(case_id)` and `get_origin(case_id, origin)`.
`FEATURE_ORIGINS_BY_TASK` records which origins appear per task.

An **opt-in predictor tool template** (`tools/predictor.py`) shows the full
no-leak wiring: enabled via `agent.predictor.enabled=true`, the MCP server
registers a `get_image_predictor` tool that loads embeddings through
`FeatureStore`, calls `run_predictor`, and returns only a compact score —
never the raw vectors. `run_predictor` receives **all** origins for the case
(MRI / biopsy / prostatectomy), so participants can use one or fuse them. It is
off by default; replace the stub with a trained head to use it.

**Source**: `src/chimera_agent_baseline/features.py`,
`src/chimera_agent_baseline/tools/predictor.py`.

## RAG (search_guidelines)

Clinical guidelines are chunked, embedded with
`google/embeddinggemma-300m`, and persisted to ChromaDB at
`resources/guidelines_db/`. The embedding model runs in a separate
process on CPU (sentence-transformers via UDS), so it does not compete
with the agent LLM for GPU memory.

To rebuild with different guidelines:

```bash
python scripts/process_guidelines.py --pdf path/to/your/guidelines.pdf
```

## Output

Each case writes a single file,
`test/output/task<N>/<case_id>/prediction.json`, which is exactly the
validated structured record:

| Field | Tasks | Purpose |
|---|---|---|
| `case_id`, `task` | 1, 2, 3 | Identifiers |
| `biopsy_decision` (bool) | 1 | The decision to be evaluated |
| `action` (1 of 4) | 2 | The decision to be evaluated |
| `months_to_recurrence` (float) | 3 | The numeric prognosis |
| `confidence` (clear/borderline/uncertain) | 1, 2 | Decision confidence |
| `variable_weights` (not_used/noted/important/decisive per variable) | 1, 2 | Reasoning capture |
| `reasoning` (≥40 chars) | 1, 2, 3 | Free-text reasoning |

`form_fill_warnings` (validation retries) are logged, not written to the
output file.
