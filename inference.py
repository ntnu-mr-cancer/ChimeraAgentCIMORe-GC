"""Grand Challenge entrypoint.

Reads inputs from ``/input`` (the same ``task<N>/agent_input/<case>/``
hierarchy used locally), runs every task present, and writes structured
predictions to ``/output/task<N>/<case>/prediction.json``. Thin wrapper around
:func:`chimera_agent_baseline.run.run_agent` that loads the same
``configs/config.yaml`` Hydra reads locally and overrides the path fields to
the GC container's mount points.

For local development, use ``make run`` instead.
"""

import asyncio
import json
import logging
import tempfile
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf

from chimera_agent_baseline.rag import start_embedding_service
from chimera_agent_baseline.run import run_agent
from chimera_agent_baseline.utils import setup_logging

log = logging.getLogger(__name__)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
CONFIG_PATH = Path("/opt/app/configs/config.yaml")
RESOURCE_PATH = Path("/opt/app/resources")
MODEL_PATH = Path("/opt/ml/model/llm/")
EMBEDDING_MODEL_PATH = Path("/opt/app/resources/embedding_model")

RUN_CHIMERA_BASELINE = True

# --- Output socket filenames per task ---------------------------------------
OUTPUT_SOCKETS = {
    1: {
        "decision": "prostate-biospy-decision.json",
        "reasoning": "prostate-biospy-decision-reasoning.json",
    },
    2: {
        "decision": "prostate-treatment-decision.json",
        "reasoning": "prostate-treatment-decision-reasoning.json",
    },
    3: {
        "decision": "prostate-time-to-recurrence-or-last-follow-up.json",
        "reasoning": "prostate-time-to-recurrence-or-last-follow-up-reasoning.json",
    },
}

# --- Fallback predictions (used if the agent crashes) ------------------------
FALLBACK_PREDICTIONS = {
    1: {
        "biopsy_decision": "no",
        "free_text": "Fallback prediction: the agent was unable to complete inference for this case.",
        "confidence": "uncertain",
        "variable_weights": {},
    },
    2: {
        "treatment_recommendation": {"primary": "active_surveillance"},
        "free_text": "Fallback prediction: the agent was unable to complete inference for this case.",
        "confidence": "uncertain",
        "variable_weights": {},
    },
    3: {
        "event": 0,
        "months_to_recurrence": 24.0,
        "free_text": "Fallback prediction: the agent was unable to complete inference for this case.",
    },
}


def run():
    setup_logging("INFO")

    try:
        interface_key = get_interface_key()

        handler = {
            (
                "prostate-biopsy-decision-clinical-data",
                "prostate-modality-level-neural-representations",
                "structured-prompt",
            ): interf0_handler,
            (
                "prostate-modality-level-neural-representations",
                "prostate-treatment-decision-clinical-data",
                "structured-prompt",
            ): interf1_handler,
            (
                "prostate-modality-level-neural-representations",
                "prostate-time-to-recurrence-or-last-follow-up-clin",
                "structured-prompt",
            ): interf2_handler,
        }.get(interface_key)

        if handler is None:
            log.error("Unknown interface key: %s", interface_key)
            _write_all_fallbacks(interface_key)
            return 0

        return handler()

    except Exception:
        log.error("Fatal error in run(): %s", traceback.format_exc())
        # Try to write some fallback output so GC doesn't get nothing
        try:
            task = _guess_task_from_key(interface_key) if "interface_key" in dir() else None
            if task is not None:
                _write_fallback_output(task)
        except Exception:
            log.error("Even fallback output failed: %s", traceback.format_exc())
        return 0


def interf0_handler():
    input_structured_prompt = load_json_file(location=INPUT_PATH / "structured-prompt.json")
    input_neural = load_json_file(location=INPUT_PATH / "prostate-modality-level-neural-representations.json")
    input_clinical = load_json_file(location=INPUT_PATH / "prostate-biopsy-decision-clinical-data.json")

    if RUN_CHIMERA_BASELINE:
        return run_baseline_for_gc_interface(
            task=1,
            structured_prompt=input_structured_prompt,
            clinical_data=input_clinical,
            neural_representations=input_neural,
        )

    # --- Dummy fallback (unused) ---
    _write_dummy_output(task=1)
    return 0


def interf1_handler():
    input_structured_prompt = load_json_file(location=INPUT_PATH / "structured-prompt.json")
    input_neural = load_json_file(location=INPUT_PATH / "prostate-modality-level-neural-representations.json")
    input_clinical = load_json_file(location=INPUT_PATH / "prostate-treatment-decision-clinical-data.json")

    if RUN_CHIMERA_BASELINE:
        return run_baseline_for_gc_interface(
            task=2,
            structured_prompt=input_structured_prompt,
            clinical_data=input_clinical,
            neural_representations=input_neural,
        )

    _write_dummy_output(task=2)
    return 0


def interf2_handler():
    input_structured_prompt = load_json_file(location=INPUT_PATH / "structured-prompt.json")
    input_neural = load_json_file(location=INPUT_PATH / "prostate-modality-level-neural-representations.json")
    input_clinical = load_json_file(
        location=INPUT_PATH / "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json"
    )

    if RUN_CHIMERA_BASELINE:
        return run_baseline_for_gc_interface(
            task=3,
            structured_prompt=input_structured_prompt,
            clinical_data=input_clinical,
            neural_representations=input_neural,
        )

    _write_dummy_output(task=3)
    return 0


# ---------------------------------------------------------------------------
# Core: run the agent and write outputs
# ---------------------------------------------------------------------------

def run_baseline_for_gc_interface(
    *,
    task: int,
    structured_prompt: dict,
    clinical_data: dict,
    neural_representations: dict,
) -> int:
    """Run the baseline from flat GC /input files and write flat /output files."""

    setup_logging("INFO")

    case_id = structured_prompt.get("case_id", "gc-case")

    clinical_filename_by_task = {
        1: "prostate-biopsy-decision-clinical-data.json",
        2: "prostate-treatment-decision-clinical-data.json",
        3: "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",
    }
    clinical_filename = clinical_filename_by_task[task]

    structured_prompt = dict(structured_prompt)
    clinical_data = dict(clinical_data)
    neural_representations = dict(neural_representations)

    structured_prompt.setdefault("case_id", case_id)
    structured_prompt.setdefault("task", task)
    clinical_data.setdefault("case_id", case_id)
    neural_representations.setdefault("case_id", case_id)

    with tempfile.TemporaryDirectory(prefix="chimera-gc-") as tmp:
        tmp_root = Path(tmp)

        internal_input_root = tmp_root / "input"
        internal_output_root = tmp_root / "output"

        case_dir = internal_input_root / f"task{task}" / "agent_input" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        write_json_file(location=case_dir / "structured-prompt.json", content=structured_prompt)
        write_json_file(location=case_dir / clinical_filename, content=clinical_data)
        write_json_file(
            location=case_dir / "prostate-modality-level-neural-representations.json",
            content=neural_representations,
        )

        cfg = load_config(
            internal_input_root=internal_input_root,
            internal_output_root=internal_output_root,
            task=task,
        )

        log.info("Running CHIMERA baseline for task=%s case_id=%s", task, case_id)

        # --- Start embedding service (optional) ---
        embed_svc = None
        try:
            embed_svc = start_embedding_service(cfg.paths.embedding_model_dir)
        except Exception:
            log.warning("Failed to start embedding service — continuing without RAG", exc_info=True)

        # --- Run the agent ---
        agent_failed = False
        try:
            asyncio.run(run_agent(cfg))
        except Exception:
            log.error("Agent failed: %s", traceback.format_exc())
            agent_failed = True
        finally:
            if embed_svc:
                try:
                    embed_svc.stop()
                except Exception:
                    pass

        # --- Read back predictions (or use fallback) ---
        if agent_failed:
            log.warning("Using fallback prediction for task %d", task)
            _write_fallback_output(task)
            return 0

        produced_dir = internal_output_root / f"task{task}" / case_id
        if not produced_dir.is_dir():
            log.warning("No output dir produced at %s — using fallback", produced_dir)
            _write_fallback_output(task)
            return 0

        prediction_files = list(produced_dir.glob("*.json"))
        if not prediction_files:
            log.warning("No prediction files found in %s — using fallback", produced_dir)
            _write_fallback_output(task)
            return 0

        # --- Copy whatever the agent produced to /output ---
        for json_file in prediction_files:
            try:
                content = load_json_file(location=json_file)
                write_json_file(location=OUTPUT_PATH / json_file.name, content=content)
            except Exception:
                log.warning("Failed to copy %s to output", json_file, exc_info=True)

        log.info("Wrote GC result sockets for task %d to %s", task, OUTPUT_PATH)
        return 0


# ---------------------------------------------------------------------------
# Fallback output helpers
# ---------------------------------------------------------------------------

def _write_fallback_output(task: int) -> None:
    """Write minimal fallback predictions so GC gets valid output sockets."""
    sockets = OUTPUT_SOCKETS[task]
    pred = FALLBACK_PREDICTIONS[task]

    if task == 1:
        decision_value = pred["biopsy_decision"]
        reasoning_value = {
            "free_text": pred["free_text"],
            "confidence": pred["confidence"],
            "variable_weights": pred["variable_weights"],
        }
    elif task == 2:
        decision_value = pred["treatment_recommendation"]["primary"]
        reasoning_value = {
            "free_text": pred["free_text"],
            "confidence": pred["confidence"],
            "variable_weights": pred["variable_weights"],
        }
    else:
        decision_value = {
            "event": pred["event"],
            "months_to_recurrence": pred["months_to_recurrence"],
        }
        reasoning_value = pred["free_text"]

    write_json_file(location=OUTPUT_PATH / sockets["decision"], content=decision_value)
    write_json_file(location=OUTPUT_PATH / sockets["reasoning"], content=reasoning_value)
    log.info("Wrote fallback output for task %d", task)


def _write_all_fallbacks(interface_key) -> None:
    """Write fallback for all tasks when we can't even detect the interface."""
    for task in (1, 2, 3):
        try:
            _write_fallback_output(task)
        except Exception:
            pass


def _guess_task_from_key(interface_key) -> int | None:
    """Best-effort task detection from interface key."""
    if "biopsy" in str(interface_key):
        return 1
    if "treatment" in str(interface_key):
        return 2
    if "recurrence" in str(interface_key):
        return 3
    return None


def _write_dummy_output(task: int) -> None:
    """Write dummy outputs (unused when RUN_CHIMERA_BASELINE=True)."""
    _write_fallback_output(task)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def load_config(internal_input_root: Path, internal_output_root: Path, task: int):
    """Load config and override paths for one GC interface run."""
    cfg = OmegaConf.load(CONFIG_PATH)

    OmegaConf.update(cfg, "paths.data_root", str(internal_input_root))
    OmegaConf.update(cfg, "paths.output_dir", str(internal_output_root))
    OmegaConf.update(cfg, "paths.resource_dir", str(RESOURCE_PATH))
    OmegaConf.update(cfg, "paths.model_dir", str(MODEL_PATH))
    OmegaConf.update(cfg, "paths.embedding_model_dir", str(EMBEDDING_MODEL_PATH))

    OmegaConf.update(cfg, "agent.tasks", [task])
    OmegaConf.update(cfg, "agent.pids", None)
    OmegaConf.update(cfg, "agent.limit", None)
    OmegaConf.update(cfg, "agent.step_timeout", 900)

    return cfg


if __name__ == "__main__":
    raise SystemExit(run())
