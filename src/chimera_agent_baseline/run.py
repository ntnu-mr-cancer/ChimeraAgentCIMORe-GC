"""Agent entry-point.

Hydra-driven runner. Walks the hierarchical input tree
``<data_root>/task<N>/agent_input/<case>/`` and runs the LangGraph ReAct +
form-fill graph on every case, one task at a time, writing
``<output_dir>/task<N>/<case>/prediction.json``.

By default every task present under ``data_root`` is run (``agent.tasks``);
missing task dirs are skipped. The model is loaded once and reused across
tasks. The Grand Challenge container uses the same layout, rooted at
``/input`` / ``/output``.

Usage::

    make run                                       # all tasks under data/
    make run RUN_ARGS="agent.tasks=[2]"            # just task 2
    make run RUN_ARGS="+experiment=qwen_local"     # swap to Qwen
    make run RUN_ARGS="agent.limit=5"              # first 5 cases per task
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm
import time
import os
import csv
from pprint import pprint


import hydra
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from omegaconf import DictConfig

from chimera_agent_baseline.agent.graph import create_graph
from chimera_agent_baseline.agent.prompts import build_system_prompt
from chimera_agent_baseline.case_loader import load_cases
from chimera_agent_baseline.models import load_model
from chimera_agent_baseline.rag import start_embedding_service
from chimera_agent_baseline.utils import setup_logging
from chimera_agent_baseline.hf_snapshot_utils import resolve_model_path


load_dotenv()
log = logging.getLogger(__name__)


_VALID_TASKS = (1, 2, 3)

def _shard_queries(
    queries: list[dict],
    worker_id: int,
    num_workers: int,
) -> list[dict]:
    """Keep only the cases assigned to this worker."""

    if num_workers <= 1:
        return queries

    return [
        q
        for idx, q in enumerate(queries)
        if idx % num_workers == worker_id
    ]

def _task_input_dir(cfg: DictConfig, task: int) -> Path:
    return Path(cfg.paths.data_root) / f"task{task}" / "agent_input"

def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2))


def _write_task_outputs(
    case_dir: Path,
    task: int,
    structured: dict[str, Any],
) -> None:
    if task == 1:
        decision = (
            "yes"
            if structured["biopsy_decision"]
            else "no"
        )

        reasoning = {
            "confidence": structured["confidence"],
            "variable_weights": structured["variable_weights"],
            "reveal_sequence": structured["reveal_sequence"],
            "free_text": structured["reasoning"],
        }

        _write_json(
            case_dir / "prostate-biopsy-decision.json",
            decision,
        )

        _write_json(
            case_dir / "prostate-biopsy-decision-reasoning.json",
            reasoning,
        )

        return

    if task == 2:
        reasoning = {
            "confidence": structured["confidence"],
            "variable_weights": structured["variable_weights"],
            "reveal_sequence": structured["reveal_sequence"],
            "free_text": structured["reasoning"],
        }

        _write_json(
            case_dir / "prostate-treatment-decision.json",
            structured["action"],
        )

        _write_json(
            case_dir / "prostate-treatment-decision-reasoning.json",
            reasoning,
        )

        return

    if task == 3:
        _write_json(
            case_dir /
            "prostate-time-to-recurrence-or-last-follow-up.json",
            {
                "event": structured.get("event", 0),
                "months_to_recurrence":
                    structured["months_to_recurrence"],
            },
        )

        _write_json(
            case_dir /
            "prostate-time-to-recurrence-or-last-follow-up-reasoning.json",
            structured["reasoning"],
        )

        return



def _run_plan(cfg: DictConfig) -> list[tuple[int, Path]]:
    """Resolve ``agent.tasks`` to ``(task_int, input_dir)`` pairs that exist."""
    plan: list[tuple[int, Path]] = []
    for raw in cfg.agent.tasks:
        task = int(raw)
        if task not in _VALID_TASKS:
            raise ValueError(f"Unknown task {task!r} in agent.tasks; expected one of {list(_VALID_TASKS)}")
        input_dir = _task_input_dir(cfg, task)
        if input_dir.is_dir():
            plan.append((task, input_dir))
        else:
            log.warning("Skipping task %d: %s not found", task, input_dir)
    if not plan:
        raise FileNotFoundError(f"No task data found under {cfg.paths.data_root} for tasks {list(cfg.agent.tasks)}")
    return plan


def _filter_queries(queries: list[dict], cfg: DictConfig) -> list[dict]:
    """Apply optional ``cfg.agent.pids`` / ``cfg.agent.limit`` subset filters."""
    pids = cfg.agent.get("pids")
    if pids:
        wanted = set(pids)
        out = [q for q in queries if q["case_id"] in wanted]
        missing = wanted - {q["case_id"] for q in out}
        if missing:
            log.warning("Requested pids not found in input dir: %s", sorted(missing))
        log.info("Filtered to %d hand-picked cases: %s", len(out), [q["case_id"] for q in out])
        return out
    limit = cfg.agent.get("limit")
    if limit:
        out = queries[: int(limit)]
        log.info("Limiting to first %d cases", len(out))
        return out
    return queries


def _mcp_args(cfg: DictConfig, input_dir: Path, registry: str) -> list[str]:
    args = [
        "-m",
        "chimera_agent_baseline.mcp_server",
        "--data-dir",
        str(input_dir),
        "--resource-dir",
        str(cfg.paths.resource_dir),
        "--tool-registry",
        registry,
    ]
    # Optional image-embedding predictor tool (off by default).
    predictor = cfg.agent.get("predictor")
    if predictor and predictor.get("enabled"):
        args += ["--enable-predictor"]
    return args


async def _run_task(
    cfg: DictConfig,
    task_int: int,
    input_dir: Path,
    model,
    system_prompt: str,
) -> int:
    """Run every case for one task; returns the number of predictions written."""
    registry = f"task{task_int}"

    queries = _filter_queries(
        load_cases(input_dir, task=task_int),
        cfg,
    )

    worker_id = int(cfg.agent.get("worker_id", 0))
    num_workers = int(cfg.agent.get("num_workers", 1))

    queries = _shard_queries(
        queries,
        worker_id=worker_id,
        num_workers=num_workers,
    )

    log.info(
        "Task %d: worker %d/%d assigned %d cases",
        task_int,
        worker_id,
        num_workers,
        len(queries),
    )

    task_dir = Path(cfg.paths.output_dir) / f"task{task_int}"
    task_dir.mkdir(parents=True, exist_ok=True)

    failed_cases_file = task_dir / "failed_cases.jsonl"

    skip_existing = cfg.agent.get("skip_existing_predictions", True)

    # if skip_existing:
    #     log.info(
    #         "Task %d: skipping cases with existing prediction.json files "
    #         "(set agent.skip_existing_predictions=false to force re-predictions)",
    #         task_int,
    #     )

    #     remaining_queries = []

    #     for query in queries:
    #         case_id = query["case_id"]
    #         case_dir = task_dir / case_id
    #         if any(case_dir.glob("*.json")):
    #             continue
    #         remaining_queries.append(query)

    #     skipped_count = len(queries) - len(remaining_queries)

    #     log.info(
    #         "Task %d: %d/%d cases already have predictions",
    #         task_int,
    #         skipped_count,
    #         len(queries),
    #     )

    #     queries = remaining_queries
        
    # skip_missing_gt = cfg.agent.get("skip_missing_gt", True)
    # gt_csv_path = Path("cases_with_gt.csv")

    # if skip_missing_gt and gt_csv_path.exists():
    #     with open(gt_csv_path, mode="r", encoding="utf-8") as f:
    #         valid_case_ids = {row["case_id"].strip() for row in csv.DictReader(f) if row.get("case_id")}

    #     remaining_queries = [q for q in queries if q.get("case_id") in valid_case_ids]
    #     skipped_count = len(queries) - len(remaining_queries)

    #     log.info("Task %d: filtered %d/%d cases lacking ground truth", task_int, skipped_count, len(queries))
    #     queries = remaining_queries

    # if not queries:
    #     log.info(
    #         "Task %d: all cases already have predictions. "
    #         "Skipping MCP startup and graph creation. "
    #         "Set agent.skip_existing_predictions=false to force re-predictions.",
    #         task_int,
    #     )
    #     return 0

    log.info(
        "Task %d: starting MCP server (data_dir=%s)",
        task_int,
        input_dir,
    )

    client = MultiServerMCPClient(
        {
            "chimera": {
                "command": sys.executable,
                "args": _mcp_args(cfg, input_dir, registry),
                "transport": "stdio",
            },
        }
    )

    tools = await client.get_tools()

    log.info(
        "Task %d: loaded %d tools from MCP server",
        task_int,
        len(tools),
    )
    
    for tool in tools:
        log.info(tool.name)
   

    graph = create_graph(
        tools,
        model,
        system_prompt,
        step_timeout=cfg.agent.step_timeout,
        form_fill_max_retries=cfg.agent.form_fill.max_retries,
    )

    n_done = 0
    n_failed = 0

    pbar = tqdm(
    queries,
    desc=f"[GPU:{worker_id} Task:{task_int}]",
    dynamic_ncols=True,
    position=worker_id,
    )

    for query in pbar:
        case_id = query["case_id"]

        try:
            initial_state: dict[str, Any] = {
                "messages": [HumanMessage(content=query["context"])],
                "case_id": case_id,
                "task": task_int,
                "patient": {
                    "psa": query.get("psa"),
                    "age": query.get("age"),
                },
            }

            result = await graph.ainvoke(
                initial_state,
                {"recursion_limit": cfg.agent.max_iterations},
            )

            structured = result["structured_response"]

            # case_dir = task_dir / case_id
            # case_dir.mkdir(parents=True, exist_ok=True)

            # prediction_file = case_dir / "prediction.json"

            # prediction_file.write_text(
            #     json.dumps(structured, indent=2)
            # )
            
            case_dir = task_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)

            _write_task_outputs(
                case_dir,
                task_int,
                structured,
            )

            n_done += 1

            pbar.set_postfix(
                failed=n_failed,
                case=case_id,
            )

        except Exception as exc:
            n_failed += 1

            log.exception(
                "Case=%s FAILED after %.1fs",
                case_id,
            )

            failure_record = {
                "case_id": case_id,
                "task": task_int,
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            with failed_cases_file.open("a") as f:
                f.write(json.dumps(failure_record) + "\n")

            log.warning(
                "Case=%s recorded in %s and skipped. Continuing.",
                case_id,
                failed_cases_file,
            )

            pbar.set_postfix(
                failed=n_failed,
                case=case_id,
            )
            continue

    log.info(
        "Task %d complete: wrote=%d failed=%d output_dir=%s",
        task_int,
        n_done,
        n_failed,
        task_dir,
    )

    if n_failed:
        log.warning(
            "Task %d had %d failed cases. See %s",
            task_int,
            n_failed,
            failed_cases_file,
        )

    return n_done


async def run_agent(cfg: DictConfig) -> None:
    """Run every task present under ``data_root`` (model loaded once, reused)."""
    plan = _run_plan(cfg)
    log.info("Run plan: tasks %s", [t for t, _ in plan])

    model = load_model(cfg)
    system_prompt = build_system_prompt()

    total = 0
    for task_int, input_dir in plan:
        total += await _run_task(cfg, task_int, input_dir, model, system_prompt)
    log.info("Done. Wrote %d predictions across %d task(s).", total, len(plan))


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    worker_id = int(cfg.agent.get("worker_id", 0))
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    setup_logging(cfg.logging.level, worker_id=worker_id)
    
    print("Usin the following config:")
    pprint(cfg)
    
    os.environ["CHIMERA_EMBED_SOCKET"] = (f"/tmp/chimera_embed_{worker_id}.sock")
    
    # Automatically downloads the model if it does not exits in HF_HOME
    # embedding_model_path = resolve_model_path(cfg.paths.embedding_model_dir)
    # embed_svc = start_embedding_service(embedding_model_path)
    embed_svc = start_embedding_service(cfg.paths.embedding_model_dir)
    try:
        asyncio.run(run_agent(cfg))
    finally:
        if embed_svc:
            embed_svc.stop()


if __name__ == "__main__":
    main()
