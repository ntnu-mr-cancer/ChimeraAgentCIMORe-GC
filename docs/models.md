# Model Configuration

The agent supports two model providers, switched via `model.provider`:

- **`vllm`** — in-process inference via [vLLM](https://docs.vllm.ai).
  Single-process, no network. This is what the Grand Challenge
  container uses.
- **`openai`** — any OpenAI-compatible HTTP endpoint (llama.cpp's
  `llama-server`, vLLM's OpenAI server, OpenAI itself). Useful for
  swapping models without rebuilding vLLM and for verifying that the
  orchestration is provider-neutral.

Both return a LangChain `BaseChatModel`; the agent graph (ReAct +
form-fill) does not care which is used.

## Default — Gemma 4 E2B-it via vLLM

```yaml
# configs/config.yaml
model:
  provider: vllm
  model_id: google/gemma-4-E2B-it
  tool_parser: gemma4
```

Download:

```bash
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('google/gemma-4-E2B-it', local_dir='model/')"
```

When `model/` is non-empty those weights are loaded; otherwise vLLM
downloads from HuggingFace using `model_id`.

## Swapping the vLLM model

```yaml
model:
  provider: vllm
  model_id: meta-llama/Llama-3.1-8B-Instruct
  tool_parser: llama
```

Download the weights, no code changes needed.

### Tool-call parsers

| `tool_parser` | Models | Format |
|---|---|---|
| `gemma4` | Gemma 4 | `call:func_name{key:value}` |
| `hermes` | Hermes, many fine-tunes | `<tool_call>{"name": ...}</tool_call>` |
| `llama` | Llama 3.x | JSON-based |
| `mistral` | Mistral, Mixtral | `[TOOL_CALLS]` + JSON |
| `pythonic` | Python-style calls | `func_name(arg=val)` |

`model.tool_parser` selects a vLLM parser-utils module —
`vllm.tool_parsers.<parser>_utils`, whose `parse_tool_calls()` the
backend calls. List the parsers available in your vLLM install:

```bash
ls "$(python -c 'import vllm.tool_parsers; print(vllm.tool_parsers.__path__[0])')"/*_utils.py
```

If the named parser module can't be loaded, the run fails loudly at the
first generation rather than silently producing no tool calls — set
`model.tool_parser` to a parser your vLLM ships, or use the
OpenAI-compatible provider.

## Using an OpenAI-compatible server

`configs/experiment/qwen_local.yaml` shows a working setup for
[llama.cpp's server](https://github.com/ggerganov/llama.cpp) serving
Qwen3 over `http://127.0.0.1:8765/v1`. Spin it up:

```bash
LLAMA_CTX=16384 ~/.local/bin/llama-start
make run RUN_ARGS="+experiment=qwen_local"
```

Minimum config:

```yaml
# @package _global_
model:
  provider: openai
  model_id: <whatever the server reports in /v1/models>
  base_url: http://127.0.0.1:8765/v1
  api_key_env: LLAMA_API_KEY        # env var with the bearer token
  request_timeout: 600
  extra_body:                       # forwarded as-is per request
    reasoning_budget_tokens: 4096   # llama.cpp soft cap on <think> length
```

`extra_body` is passed straight to `ChatOpenAI` and into the request
body, so any provider-specific knob (e.g. `reasoning_budget_tokens`,
`chat_template_kwargs`) works.

### Reasoning content propagation

`langchain_openai`'s default `ChatOpenAI` discards the
`reasoning_content` field that providers like llama.cpp / Qwen3 put on
the assistant message. The `ChatOpenAICompat` subclass at
`src/chimera_agent_baseline/models/openai_compat.py` copies it onto
`AIMessage.additional_kwargs["reasoning_content"]`, so the chain of
thought survives on the message history. The baseline does not persist
it — `prediction.json` holds only the validated structured record — but
it's there on the messages if you want to capture a reasoning trace in
`run.py`.

## Experiment overlays

```yaml
# configs/experiment/llama8b.yaml
# @package _global_
model:
  provider: vllm
  model_id: meta-llama/Llama-3.1-8B-Instruct
  tool_parser: llama
generation:
  temperature: 0.6
```

```bash
make run RUN_ARGS="+experiment=llama8b"
```

## Embedding model (RAG)

The `search_guidelines` tool runs a local RAG pipeline (see
[architecture.md](architecture.md)): a ChromaDB vector store of guideline
chunks, queried with embeddings from `google/embeddinggemma-300m`. Two
artifacts must be present under `resources/` for it to return real results:

| Artifact | Size | Shipped how |
|---|---|---|
| `resources/guidelines_db/` | ~20 MB | **Committed to the repo** — pre-built, use as-is |
| `resources/embedding_model/` | ~1.2 GB | **Gitignored** — you download it |

The Grand Challenge container has no network at runtime, and the Dockerfile
bakes the whole `resources/` tree into the image (`COPY resources/ …`). So
the embedding model must be on disk **before** `make gc-build`, or
`search_guidelines` returns empty results (the embedding service silently
skips startup when `resources/embedding_model/` is missing).

### Download the model (recommended — keeps the pre-built DB)

Accept the license on
[embeddinggemma](https://huggingface.co/google/embeddinggemma-300m) and set
`HF_TOKEN` (e.g. in `.env`), then:

```bash
make fetch-embedding-model        # → scripts/download_embedding_model.py
```

This fetches only the model into `resources/embedding_model/` and leaves the
committed DB untouched. It's the path participants want: no PDF, no
re-embedding. The model is saved in the same `SentenceTransformer` on-disk
format the runtime loads, so it stays compatible with the committed DB.

### Rebuild the corpus from the PDF (maintainer-only)

```bash
make process-guidelines           # → scripts/process_guidelines.py
```

This re-extracts and re-chunks `docs/internal/guidelines.pdf`, re-embeds
everything, and writes **both** `guidelines_db/` and `embedding_model/`. It
requires the source PDF, which is **not** shipped in the repo — so it's for
maintainers updating the knowledge base, not for participants. Only reach for
it if you're changing the corpus or chunking; otherwise prefer
`make fetch-embedding-model`.

> If you swap the embedding model, you must rebuild the DB with the same
> model — the committed `guidelines_db/` only matches `embeddinggemma-300m`
> (query and document embeddings must come from one model).

## Choosing a model

- **Tool calling** is required by the ReAct loop. The form-fill node
  does **not** use function calling — it goes through prompt +
  `PydanticOutputParser`, so even models with weak tool calling work
  for that step.
- **GPU memory**: Grand Challenge offers T4 (16 GB) or A10G (24 GB).
  Gemma 4 E2B (≈ 10 GB bf16) fits comfortably; larger models may need
  quantisation.
- **vLLM support**: verify the architecture is in vLLM's
  [supported list](https://docs.vllm.ai/en/latest/models/supported_models.html).
