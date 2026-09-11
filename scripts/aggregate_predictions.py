"""Rebuild the Grand Challenge ``predictions.json`` from per-case job outputs.

On Grand Challenge each case is a separate algorithm job and the **platform**
assembles every job's inputs + result sockets into one ``predictions.json``.
Locally, ``do_test_run.sh`` runs each interface/case one-by-one (writing its
result sockets under ``test/outputs/<interface>/<case>/``); this script performs
the same aggregation afterwards so the local test mirror matches the platform.

One entry per case is emitted, in the GC shape::

    {
      "pk": <deterministic hash of <interface>/<case>>,
      "inputs":  [<the case's input sockets, echoed from inputs.json>],
      "outputs": [<the case's result sockets, values read back from disk>],
      "exec_duration": null,
      "invoke_duration": null,
      "status": "Succeeded" | "Failed"
    }

The tool ``reveal_sequence`` is carried inside the reasoning socket's value
(written there by the container), so it is preserved in ``predictions.json``
without any extra top-level keys.

Stdlib-only, so it runs on the host with any ``python3`` (no project deps).

Usage::

    python3 scripts/aggregate_predictions.py --input test/input --output test/outputs
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

# The clinical-data socket slug identifies the interface / task (GC truncates
# slugs to 50 chars, hence the clipped task-3 slug).
CLINICAL_SLUG_TO_TASK: dict[str, int] = {
    "prostate-biopsy-decision-clinical-data": 1,
    "prostate-treatment-decision-clinical-data": 2,
    "prostate-time-to-recurrence-or-last-follow-up-clin": 3,
}

_KIND_FLAGS: dict[str, bool] = {
    "is_image_kind": False,
    "is_panimg_kind": False,
    "is_dicom_image_kind": False,
    "is_json_kind": True,
    "is_file_kind": False,
}

# Static ``example_value`` metadata for each result socket, mirroring the GC
# interface export (fixed per interface, independent of the actual case).
_TASK1_REASONING_EXAMPLE: dict[str, Any] = {
    "free_text": (
        "The decision is driven by three critical factors: the PI-RADS 5 score, the extremely high "
        "csPCa predicted probability (0.96), and the frankly elevated PSA level (187.0 ng/mL) with a "
        "rapid upward trend."
    ),
    "confidence": "clear",
    "variable_weights": {
        "bx": "decisive",
        "fh": "noted",
        "age": "important",
        "dre": "noted",
        "psa": "noted",
        "vol": "noted",
        "psad": "not_used",
        "cspca": "not_used",
        "pirads": "important",
        "comorbidity": "noted",
    },
    "reveal_sequence": [],
}
_TASK2_REASONING_EXAMPLE: dict[str, Any] = {
    "free_text": (
        "The decision to proceed to active treatment is driven primarily by the confluence of "
        "high-grade pathology (Gleason 4+4 with PNI), high-risk imaging findings (PI-RADS 5 with "
        "seminal vesicle invasion), and aggressive biochemical progression indicated by the rapid "
        "PSA escalation."
    ),
    "confidence": "clear",
    "variable_weights": {
        "ct": "important",
        "fh": "noted",
        "age": "important",
        "psa": "important",
        "psad": "noted",
        "cspca": "noted",
        "pirads": "decisive",
        "bx_isup": "decisive",
        "bx_gl_sec": "noted",
        "bx_gl_prim": "noted",
        "comorbidity": "noted",
    },
    "reveal_sequence": [],
}
_TASK3_REASONING_EXAMPLE = (
    "High-risk pathology (lymph node metastasis, pT4b), positive surgical margins, and seminal "
    "vesicle invasion drive the prediction. These aggressive features significantly elevate the "
    "immediate risk of biochemical recurrence post-radical prostatectomy."
)
_TASK3_DECISION_EXAMPLE: dict[str, Any] = {"event": 0, "months_to_recurrence": 31.4}

# Result sockets per task, in the order they appear in the GC ``predictions.json``
# (tasks 1 & 2: decision then reasoning; task 3: reasoning then decision).
OUTPUT_SOCKETS: dict[int, list[dict[str, Any]]] = {
    1: [
        {
            "slug": "prostate-biospy-decision",
            "relative_path": "prostate-biospy-decision.json",
            "example_value": "yes",
        },
        {
            "slug": "prostate-biospy-decision-reasoning",
            "relative_path": "prostate-biospy-decision-reasoning.json",
            "example_value": _TASK1_REASONING_EXAMPLE,
        },
    ],
    2: [
        {
            "slug": "prostate-treatment-decision",
            "relative_path": "prostate-treatment-decision.json",
            "example_value": "watchful_waiting",
        },
        {
            "slug": "prostate-treatment-decision-reasoning",
            "relative_path": "prostate-treatment-decision-reasoning.json",
            "example_value": _TASK2_REASONING_EXAMPLE,
        },
    ],
    3: [
        {
            "slug": "prostate-time-to-recurrence-or-last-follow-up-reas",
            "relative_path": "prostate-time-to-recurrence-or-last-follow-up-reasoning.json",
            "example_value": _TASK3_REASONING_EXAMPLE,
        },
        {
            "slug": "prostate-time-to-recurrence-or-last-follow-up",
            "relative_path": "prostate-time-to-recurrence-or-last-follow-up.json",
            "example_value": _TASK3_DECISION_EXAMPLE,
        },
    ],
}


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _detect_task(inputs: list[dict[str, Any]]) -> int:
    slugs = {sv["socket"]["slug"] for sv in inputs}
    for slug, task in CLINICAL_SLUG_TO_TASK.items():
        if slug in slugs:
            return task
    raise ValueError(f"No known clinical-data socket in inputs.json (got {sorted(slugs)})")


def _pk_for(interface_id: str, case_id: str) -> str:
    """Deterministic pk — a stable hash of the ``<interface>/<case>`` identity."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chimera/{interface_id}/{case_id}"))


def _build_output_sockets(task: int, case_out_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read the written result sockets back into ``predictions.json`` shape.

    Returns the socket list and whether every socket file was found on disk.
    """
    sockets: list[dict[str, Any]] = []
    complete = True
    for spec in OUTPUT_SOCKETS[task]:
        socket_file = case_out_dir / spec["relative_path"]
        if socket_file.exists():
            value: Any = _load_json(socket_file)
        else:
            value = None
            complete = False
        sockets.append(
            {
                "socket": {
                    "slug": spec["slug"],
                    "relative_path": spec["relative_path"],
                    "example_value": spec["example_value"],
                    **_KIND_FLAGS,
                },
                "file": None,
                "image": None,
                "value": value,
            }
        )
    return sockets, complete


def aggregate(input_root: Path, output_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for inputs_path in sorted(input_root.rglob("inputs.json")):
        case_dir = inputs_path.parent
        inputs = _load_json(inputs_path)
        task = _detect_task(inputs)

        parts = case_dir.relative_to(input_root).parts
        if len(parts) >= 2:
            interface_id, case_id = parts[-2], parts[-1]
        elif len(parts) == 1:
            interface_id, case_id = parts[0], parts[0]
        else:
            interface_id, case_id = f"task{task}", "case"

        case_out_dir = output_root / Path(*parts) if parts else output_root
        outputs, complete = _build_output_sockets(task, case_out_dir)

        entries.append(
            {
                "pk": _pk_for(interface_id, case_id),
                "inputs": inputs,
                "outputs": outputs,
                "exec_duration": None,
                "invoke_duration": None,
                "status": "Succeeded" if complete else "Failed",
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-case GC job outputs into predictions.json")
    parser.add_argument("--input", type=Path, default=Path("test/input"), help="Input root (holds inputs.json files)")
    parser.add_argument("--output", type=Path, default=Path("test/output"), help="Output root (per-case result dirs)")
    args = parser.parse_args()

    entries = aggregate(args.input, args.output)
    predictions_path = args.output / "predictions.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(entries, indent=4))
    print(f"Wrote {len(entries)} entrie(s) to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
