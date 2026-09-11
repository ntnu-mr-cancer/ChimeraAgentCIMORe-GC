#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path("experiment_results")


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten nested dictionaries."""
    items = {}

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep))
        else:
            items[new_key] = value

    return items


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_numeric_columns(df, decimals=2):
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    return df


def main():
    all_rows = []

    # ------------------------------------------------------------------
    # Collect task-level results
    # ------------------------------------------------------------------

    for model_dir in ROOT.iterdir():

        if not model_dir.is_dir():
            continue

        run_details_file = model_dir / "run_details.yaml"

        if not run_details_file.exists():
            continue

        run_details = load_yaml(run_details_file)

        results_dir = model_dir / "results"

        if not results_dir.exists():
            continue

        for task_name in ["task1", "task2", "task3"]:

            metrics_file = (
                results_dir / task_name / "aggregate_metrics.json"
            )

            if not metrics_file.exists():
                print(
                    f"Skipping {model_dir.name}/{task_name}: "
                    "missing aggregate_metrics.json"
                )
                continue

            metrics = flatten_dict(load_json(metrics_file))
            # Remove unflattened metric
            metrics.pop("decision_classification_report", None)

            row = {
                "experiment_id": model_dir.name,
                "task": task_name,
                **run_details,
                **metrics,
            }

            all_rows.append(row)

    if not all_rows:
        print("No results found.")
        return

    # ------------------------------------------------------------------
    # All results CSV
    # ------------------------------------------------------------------

    all_results_df = pd.DataFrame(all_rows)

    all_results_df = (
        all_results_df.sort_values(
            by=["experiment_id", "task"]
        ).reset_index(drop=True)
    )

    all_results_df = round_numeric_columns(all_results_df, decimals=2)

    all_results_csv = ROOT / "all_results.csv"

    all_results_df.to_csv(all_results_csv, index=False)

    print(f"Saved {all_results_csv}")

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------

    summary_rows = []

    for experiment_id, group in all_results_df.groupby("experiment_id"):

        first_row = group.iloc[0]

        summary_row = {
            "experiment_id": experiment_id,
            "model_id": first_row.get("model_id"),
            "user": first_row.get("user"),
            "date": first_row.get("date"),
            "server": first_row.get("server"),
            "engine": first_row.get("engine"),
            "docker": first_row.get("docker"),
            "notes": first_row.get("notes"),
            "task1_ranking_score": None,
            "task2_ranking_score": None,
            "task3_ranking_score": None,
        }

        for _, row in group.iterrows():
            task = row["task"]

            if task in ["task1", "task2", "task3"]:
                summary_row[
                    f"{task}_ranking_score"
                ] = row.get("ranking_score")

        summary_row["mean_ranking_score"] = (
            group["ranking_score"].mean()
        )

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df[
        [
            "experiment_id",
            "model_id",
            "user",
            "date",
            "server",
            "engine",
            "docker",
            "task1_ranking_score",
            "task2_ranking_score",
            "task3_ranking_score",
            "mean_ranking_score",
            "notes",
        ]
    ]

    summary_df = round_numeric_columns(summary_df, decimals=2)

    summary_csv = ROOT / "summary.csv"

    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved {summary_csv}")


if __name__ == "__main__":
    main()
