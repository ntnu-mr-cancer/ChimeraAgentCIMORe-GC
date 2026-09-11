# User Manual

Reference companion to the [README](../README.md). The README walks you
through quick start, I/O, and Docker; this page covers requirements,
the full project layout, and troubleshooting.

## Requirements

| Requirement | Why |
|---|---|
| Python 3.12+ | Language runtime |
| [uv](https://docs.astral.sh/uv/) | Package manager |
| NVIDIA GPU ≥ 24 GB VRAM | vLLM in-process inference (Gemma 4 ≈ 10 GB weights + KV cache + CUDA graphs) |
| CUDA 12.x + drivers | Required by vLLM |
| vLLM (`make install-vllm`) | In-process backend for local `make run` — GPU-only, not in `make install` |
| Docker + NVIDIA Container Toolkit | For `make gc-test` |
| ~15 GB disk | Model weights + guidelines DB + embedding model |
| HuggingFace token | For gated models (Gemma 4, embeddinggemma) |
| `make` | Build automation |

## Useful commands beyond the README

```bash
make run RUN_ARGS="agent.tasks=[2]"                         # run a single task
make run RUN_ARGS="+experiment=qwen_local"                  # swap LLM via overlay
make lint
make format
make gc-save                                                # export image + model tarballs
```

First `make run` takes ~30 s for vLLM to compile CUDA graphs (cached
after).

## Project layout

```
src/chimera_agent_baseline/
├── agent/
│   ├── graph.py              ReAct + form-fill wiring (~85 LoC)
│   ├── form_fill.py          Terminal node: prompt + parse against output schema
│   └── prompts.py            System prompt (modify to change agent behaviour)
├── output/
│   └── schema.py             Pydantic submission contract (Task1Output / Task2Output / Task3Output)
├── models/
│   ├── __init__.py           Provider switch (vllm / openai)
│   ├── vllm_offline.py       vLLM in-process wrapper
│   └── openai_compat.py      ChatOpenAI subclass that surfaces reasoning_content
├── tools/
│   ├── base.py               ToolSpec + CaseDataStore
│   ├── definitions.py        TASK1_TOOLS / TASK2_TOOLS / TASK3_TOOLS (add custom tools here)
│   └── predictor.py          Optional image-embedding predictor tool (off by default)
├── features.py               FeatureStore — loads per-case features.json embeddings
├── case_loader.py            Reads <case>/structured-prompt.json into agent queries
├── mcp_server.py             MCP server (per-task tool registries)
├── rag.py                    Guidelines search (ChromaDB + embeddings)
├── run.py                    Local Hydra entry-point
└── utils.py                  Logging

configs/config.yaml           Defaults (paths, model, generation, agent)
configs/experiment/           Overlays via +experiment=<name>
templates/prompts/            Jinja templates for the agent prompt
resources/                    Guidelines DB + embedding model + GC config copy
data/task{1,2,3}/agent_input/<case>/{structured-prompt,*-clinical-data,prostate-modality-level-neural-representations}.json  Per-case inputs
inference.py                  Grand Challenge container entrypoint
```

## What NOT to modify

See the README's "What NOT to change" table — one locked file:
`output/schema.py` (the submission schema). Only the final structured
output is evaluated and tool use is not scored, so everything else —
including `inference.py`, the tools, and the agent graph — is fair game.

## Troubleshooting

- **CUDA OOM / "Free memory ... less than desired GPU memory utilization"**:
  vLLM reserves a fraction of total VRAM (default 0.9), which is too aggressive
  on a 24 GB consumer card — especially if a desktop session holds a few GB.
  Lower it: `make run RUN_ARGS="generation.gpu_memory_utilization=0.80"` (the
  same key works for `gc-test` via `configs/config.yaml`). If it still OOMs,
  also lower `generation.max_model_len` (default 32768) — try 8192.
- **`search_guidelines` returns empty results** (or logs "skipping embedding
  service" / "Embedding service not running"): the embedding model is
  missing. Run `make fetch-embedding-model` (needs `HF_TOKEN`). The
  guidelines DB itself ships pre-built in `resources/guidelines_db/`; see
  [models.md → Embedding model (RAG)](models.md#embedding-model-rag).
- **HF 403 Forbidden**: accept the model license on HuggingFace.
- **vLLM compile takes long on first run**: ~30 s, cached in `~/.cache/vllm/`.
- **`make gc-test` permission error on `/output`** (`Permission denied`
  removing a previous run's `test/output/task*/...`): the container writes
  outputs as its non-root user (UID 999), so your host user can't delete the
  nested files on the next run. `make gc-test` now cleans them inside a root
  container automatically; to clear them by hand:
  `docker run --rm --user 0 -v $PWD/test/output:/o alpine rm -rf /o/*`
  (or `sudo rm -rf test/output/*`).
- **Form-fill validation fails repeatedly**: the run aborts with a
  `RuntimeError` naming the failed case and the per-attempt validation
  errors (logged, not written to the output file). Common causes are
  weak instruction-following on small models — try
  `+experiment=qwen_local` or raise `agent.form_fill.max_retries`.
