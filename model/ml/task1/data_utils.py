import os
import json
import warnings

import pandas as pd

from tqdm import tqdm
import re

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


warnings.filterwarnings("ignore", category=FutureWarning)


def load_dataset(data_path):

    rows = []

    for case_id in tqdm(os.listdir(data_path)):

        case_dir = os.path.join(data_path, case_id)

        structured_file = os.path.join(
            case_dir,
            "structured-prompt.json"
        )

        target_file = os.path.join(
            case_dir,
            "prostate-biopsy-decision.json"
        )

        if not os.path.exists(target_file):
            continue

        try:
            with open(structured_file) as f:
                structured = json.load(f)

            with open(target_file) as f:
                target = json.load(f)

        except Exception as e:
            print(f"Skipping {case_id}: {e}")
            continue


        vitals = structured.get("vitals", {})

        smoking = vitals.get("smoking", "")

        pack_years_match = re.search(
            r"(\d+(?:\.\d+)?)\s*pack",
            str(smoking)
        )

        pack_years = (
            float(pack_years_match.group(1))
            if pack_years_match
            else None
        )

        row = {
            "case_id": structured.get("case_id"),
            "age": structured.get("age"),
            "psa": structured.get("psa"),
            "psap": structured.get("psap"),
            "psav": structured.get("psav"),
            "psad": structured.get("psad"),
            "vol": structured.get("vol"),
            "pirads": structured.get("pirads"),
            "dre": structured.get("dre"),
            "ct": structured.get("ct"),
            "cspca": structured.get("cspca"),
            "bx": structured.get("bx"),

            # New features
            "bmi": pd.to_numeric(
                vitals.get("bmi"),
                errors="coerce"
            ),
            "pack_years": pack_years,

            "n_comorbidities": len(
                structured.get("pmhx", [])
            )
        }


        if isinstance(target, dict):
            row["target"] = (
                target.get("recommendation")
                or target.get("decision")
                or target.get("biopsy_recommendation")
            )
        else:
            row["target"] = target

        rows.append(row)

    df = pd.DataFrame(rows)
    df["target"] = df["target"].map({
        "yes": 1,
        "no": 0
    })

    return df


def build_preprocessor(NUM_COLS, CAT_COLS):

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="missing"
                        )
                    ),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1
                        )
                    )
                ]),
                CAT_COLS
            ),
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    )
                ]),
                NUM_COLS
            )
        ]
    )