import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Supports both:
# - Local development (defaults to repository data/)
# - Docker development image (/opt/shared/data)

DATA_ROOT = Path(
    os.getenv(
        "CHIMERA_DATA_ROOT",
        ROOT / "data",
    )
)

from inference import (
    _decision_value,
    _reasoning_value,
)

# =============================================================================
# Task configuration
# =============================================================================

TASK_CONFIG = {
    1: {
        "task_name": "task1",
        "decision_file": "prostate-biopsy-decision.json",
        "reasoning_file": "prostate-biopsy-decision-reasoning.json",
        "clinical_file": "prostate-biopsy-decision-clinical-data.json",
        "decision_output": "prostate-biospy-decision.json",
        "reasoning_output": "prostate-biospy-decision-reasoning.json",
        "decision_slug": "prostate-biospy-decision",
        "reasoning_slug": "prostate-biospy-decision-reasoning",
        "clinical_slug": "prostate-biopsy-decision-clinical-data",
        "gt_task": "urologist_biopsy_decision_cot",
        "schema_version": "1.4",
    },
    2: {
        "task_name": "task2",
        "decision_file": "prostate-treatment-decision.json",
        "reasoning_file": "prostate-treatment-decision-reasoning.json",
        "clinical_file": "prostate-treatment-decision-clinical-data.json",
        "decision_output": "prostate-treatment-decision.json",
        "reasoning_output": "prostate-treatment-decision-reasoning.json",
        "decision_slug": "prostate-treatment-decision",
        "reasoning_slug": "prostate-treatment-decision-reasoning",
        "clinical_slug": "prostate-treatment-decision-clinical-data",
        "gt_task": "urologist_treatment_decision_cot",
        "schema_version": "1.0",
    },
    3: {
    "task_name": "task3",

    "decision_file":
        "prostate-time-to-recurrence-or-last-follow-up.json",

    "reasoning_file": None,

    "clinical_file":
        "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",

    "decision_output":
        "prostate-time-to-recurrence-or-last-follow-up.json",

    "reasoning_output":
        "prostate-time-to-recurrence-or-last-follow-up-reasoning.json",

    "decision_slug":
        "prostate-time-to-recurrence-or-last-follow-up",

    "reasoning_slug":
        "prostate-time-to-recurrence-or-last-follow-up-reas",

    "clinical_slug":
        "prostate-time-to-recurrence-or-last-follow-up-clin",

    "gt_task":
        "urologist_recurrence_prognosis_cot",

    "schema_version": "1.0",
},
}

# =============================================================================
# Ground truth builders
# =============================================================================


def build_ground_truth(
    task: int,
    cfg: dict,
    case_id: str,
    decision,
    reasoning: dict,
):
    if task == 1:
        return {
            "source": "Generated from challenge dataset",
            "schema_version": cfg["schema_version"],
            "biopsy_decision": [
                {
                    "schema_version": cfg["schema_version"],
                    "task": cfg["gt_task"],
                    "case_type": "dataset",
                    "case_id": case_id,
                    "biopsy_decision": decision,
                    "confidence": reasoning.get("confidence"),
                    "variable_weights": reasoning.get(
                        "variable_weights",
                        {},
                    ),
                    "free_text": reasoning.get(
                        "free_text",
                        "",
                    ),
                    "reveal_sequence": reasoning.get(
                        "reveal_sequence",
                        [],
                    ),
                }
            ],
        }

    if task == 2:
        return {
            "source": "Generated from challenge dataset",
            "schema_version": cfg["schema_version"],
            "treatment_decision": [
                {
                    "schema_version": cfg["schema_version"],
                    "task": cfg["gt_task"],
                    "case_type": "dataset",
                    "case_id": case_id,
                    "treatment_recommendation": {
                        "primary": decision,
                    },
                    "confidence": reasoning.get("confidence"),
                    "variable_weights": reasoning.get(
                        "variable_weights",
                        {},
                    ),
                    "free_text": reasoning.get(
                        "free_text",
                        "",
                    ),
                    "reveal_sequence": reasoning.get(
                        "reveal_sequence",
                        [],
                    ),
                }
            ],
        }
    if task == 3:
        return decision


    raise NotImplementedError(
        f"Task {task} not yet supported"
    )


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tasks",
    nargs="+",
    required=True,
    type=int,
    choices=[1, 2, 3],
)

args = parser.parse_args()

EVAL_ROOT = ROOT / "eval_data"

# =============================================================================
# Section mapping
# =============================================================================

SECTION_MAPPING_SRC = (
    ROOT
    / "evaluation"
    / "ground_truth"
    / "section_variable_mapping.json"
)

if not SECTION_MAPPING_SRC.exists():
    raise FileNotFoundError(
        f"Missing section mapping file: "
        f"{SECTION_MAPPING_SRC}"
    )

# =============================================================================
# Main loop
# =============================================================================

for TASK in args.tasks:

    if TASK not in TASK_CONFIG:
        raise NotImplementedError(
            f"Task {TASK} not yet implemented"
        )

    CFG = TASK_CONFIG[TASK]
    
    
    DATASET_ROOT = (
    DATA_ROOT
    / CFG["task_name"]
    / "agent_input"
    )

    PREDICTION_ROOT = (
        ROOT
        / "test"
        / "output"
        / CFG["task_name"]
    )

    if not PREDICTION_ROOT.exists():
        raise RuntimeError(
            f"No predictions found in "
            f"{PREDICTION_ROOT}\n\n"
            f"Generate Task {TASK} predictions first."
        )
    
    # Flatten ground truth directory so it maps directly to eval_data/ground_truth/taskX
    GROUND_TRUTH_ROOT = EVAL_ROOT / "ground_truth"

    GROUND_TRUTH_DST = GROUND_TRUTH_ROOT / CFG["task_name"]

    INPUT_DST = EVAL_ROOT / CFG["task_name"] / "input"

    # -------------------------------------------------------------------------
    # Section mapping
    # -------------------------------------------------------------------------

    GROUND_TRUTH_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        SECTION_MAPPING_SRC,
        GROUND_TRUTH_ROOT
        / "section_variable_mapping.json",
    )

    print(
        f"\nPreparing {CFG['task_name']}"
    )

    # -------------------------------------------------------------------------
    # Clean previous outputs
    # -------------------------------------------------------------------------

    if GROUND_TRUTH_DST.exists():
        shutil.rmtree(GROUND_TRUTH_DST)

    if INPUT_DST.exists():
        shutil.rmtree(INPUT_DST)

    GROUND_TRUTH_DST.mkdir(
        parents=True,
        exist_ok=True,
    )

    INPUT_DST.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # Step 1: Ground truth
    # =========================================================================

    valid_cases = set()

    for case_dir in DATASET_ROOT.iterdir():

        if not case_dir.is_dir():
            continue

        decision_file = (
            case_dir
            / CFG["decision_file"]
        )

        if not decision_file.exists():
            continue

        if (
            CFG["reasoning_file"] is not None
            and not (
                case_dir
                / CFG["reasoning_file"]
            ).exists()
        ):
            continue

        case_id = case_dir.name

        valid_cases.add(case_id)

        case_output_dir = (
            GROUND_TRUTH_DST
            / case_id
        )

        case_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Tasks 1 and 2:
        # Copy original ground-truth files directly since
        # the evaluator expects the challenge file layout.
        if TASK in (1, 2):

            shutil.copy2(
                case_dir / CFG["decision_file"],
                case_output_dir / CFG["decision_file"],
            )

            shutil.copy2(
                case_dir / CFG["reasoning_file"],
                case_output_dir / CFG["reasoning_file"],
            )

        # Task 3:
        # Generate evaluator-compatible ground truth.
        else:

            with open(decision_file) as f:
                decision = json.load(f)

            gt_record = build_ground_truth(
                task=TASK,
                cfg=CFG,
                case_id=case_id,
                decision=decision,
                reasoning={},
            )

            with open(
                case_output_dir / CFG["decision_file"],
                "w",
            ) as f:
                json.dump(
                    gt_record,
                    f,
                    indent=2,
                )

    print(
        f"Generated ground truth for "
        f"{len(valid_cases)} cases"
    )

    # =========================================================================
    # Step 2: Build evaluator input package
    # =========================================================================

    jobs = []

    pk_map = {
        "source": str(
            INPUT_DST
            / "predictions.json"
        ),
        "pk_to_case": {},
    }

    for case_id in sorted(valid_cases):

        pred_file = (
            PREDICTION_ROOT
            / case_id
            / "prediction.json"
        )

        if not pred_file.exists():
            print(
                f"Missing prediction for "
                f"{case_id}"
            )
            continue

        source_case_dir = (
            DATASET_ROOT
            / case_id
        )

        prompt_file = (
            source_case_dir
            / "structured-prompt.json"
        )

        clinical_file = (
            source_case_dir
            / CFG["clinical_file"]
        )

        with open(prompt_file) as f:
            prompt_data = json.load(f)

        with open(clinical_file) as f:
            clinical_data = json.load(f)

        with open(pred_file) as f:
            prediction = json.load(f)

        decision_value = (
            _decision_value(
                TASK,
                prediction,
            )
        )

        reasoning_value = (
            _reasoning_value(
                TASK,
                prediction,
            )
        )

        pk = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{CFG['task_name']}:{case_id}",
            )
        )

        pk_map["pk_to_case"][pk] = (
            f"{CFG['task_name']}/{case_id}"
        )

        job_dir = INPUT_DST / pk

        job_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            job_dir
            / CFG["decision_output"],
            "w",
        ) as f:
            json.dump(
                decision_value,
                f,
                indent=2,
            )

        with open(
            job_dir
            / CFG["reasoning_output"],
            "w",
        ) as f:
            json.dump(
                reasoning_value,
                f,
                indent=2,
            )

        jobs.append(
            {
                "pk": pk,
                "inputs": [
                    {
                        "socket": {
                            "slug": "structured-prompt",
                            "relative_path":
                                "structured-prompt.json",
                        },
                        "file": None,
                        "image": None,
                        "value": prompt_data,
                    },
                    {
                        "socket": {
                            "slug":
                                "prostate-modality-level-neural-representations",
                            "relative_path":
                                "prostate-modality-level-neural-representations.json",
                        },
                        "file": None,
                        "image": None,
                        "value": {},
                    },
                    {
                        "socket": {
                            "slug":
                                CFG["clinical_slug"],
                            "relative_path":
                                CFG["clinical_file"],
                        },
                        "file": None,
                        "image": None,
                        "value": clinical_data,
                    },
                ],
                "outputs": [
                    {
                        "socket": {
                            "slug":
                                CFG["decision_slug"],
                            "relative_path":
                                CFG["decision_output"],
                        }
                    },
                    {
                        "socket": {
                            "slug":
                                CFG["reasoning_slug"],
                            "relative_path":
                                CFG["reasoning_output"],
                        }
                    },
                ],
            }
        )

    with open(
        INPUT_DST
        / "predictions.json",
        "w",
    ) as f:
        json.dump(
            jobs,
            f,
            indent=2,
        )

    with open(
        INPUT_DST
        / "pk_hash_to_case_map.json",
        "w",
    ) as f:
        json.dump(
            pk_map,
            f,
            indent=2,
        )

    print(
        f"Created evaluation package for "
        f"{len(jobs)} cases"
    )

print("\nDone.")