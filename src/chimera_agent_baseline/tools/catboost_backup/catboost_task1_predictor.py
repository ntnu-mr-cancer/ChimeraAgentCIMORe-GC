"""
Task 1 CatBoost predictor.

Returns:
- csPCa probability
- biopsy recommendation
- model metadata
- global feature importance
- local SHAP explanation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

PREDICTOR_TOOL_NAME = "get_biopsy_risk_prediction"

MODEL_BASE_DIR = Path("/opt/ml/model")
MODEL_DIR = MODEL_BASE_DIR / "ml" / "task1" / "artifacts" / "catboost"

FEATURES = [
    "pirads",
    "bx",
    "psa",
    "age",
    "psad",
]

import numpy as np

def make_json_safe(obj):
    """
    Recursively convert numpy/pandas objects into JSON-serializable types.
    """

    if isinstance(obj, dict):
        return {
            k: make_json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_safe(v)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            make_json_safe(v)
            for v in obj
        )

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


class Task1CatBoostPredictor:

    def __init__(self):

        self.models = [
            joblib.load(MODEL_DIR / f"fold_{i}.joblib")
            for i in range(5)
        ]

        self.case_to_fold = json.loads(
            (MODEL_DIR / "case_to_fold.json").read_text()
        )

        self.metadata = json.loads(
            (MODEL_DIR / "metadata.json").read_text()
        )

        self.global_feature_importance = (
            self.metadata.get(
                "feature_importance",
                {}
            )
        )

        self.cv_auc = self.metadata.get(
            "cv_auc",
            None
        )

        self.threshold = self.metadata.get(
            "threshold",
            0.56
        )

    def _build_feature_row(
        self,
        structured_prompt: dict
    ) -> pd.DataFrame:
        

        row = {
            "pirads": structured_prompt.get("pirads"),
            "bx": structured_prompt.get("bx"),
            "psa": structured_prompt.get("psa"),
            "age": structured_prompt.get("age"),
            "psad": structured_prompt.get("psad"),
        }

        return pd.DataFrame([row])

    def _predict_single_model(
        self,
        model,
        X: pd.DataFrame
    ) -> float:

        return float(
            model.predict_proba(X)[0, 1]
        )

    def _compute_shap(
        self,
        model,
        X: pd.DataFrame
    ) -> tuple[list[dict[str, Any]], float]:

        prep = model.named_steps["prep"]
        catboost = model.named_steps["model"]

        X_transformed = prep.transform(X)

        explainer = shap.TreeExplainer(
            catboost
        )

        shap_values = explainer.shap_values(
            X_transformed
        )

        expected_value = float(
            np.asarray(
                explainer.expected_value
            ).ravel()[0]
        )

        impacts = []

        for feature, value, effect in zip(
            FEATURES,
            X.iloc[0].values,
            shap_values[0]
        ):

            impacts.append(
                {
                    "feature": feature,
                    "value": (
                        None
                        if pd.isna(value)
                        else value
                    ),

                    "impact": round(
                        float(effect),
                        4
                    ),
                    "direction": (
                        "increases_risk"
                        if effect > 0
                        else "decreases_risk"
                    )
                }
            )

        impacts = sorted(
            impacts,
            key=lambda x: abs(
                x["impact"]
            ),
            reverse=True
        )

        return impacts, expected_value

    def _ensemble_shap(
        self,
        X: pd.DataFrame
    ) -> tuple[list[dict[str, Any]], float]:

        all_shaps = []
        baseline_values = []

        for model in self.models:

            impacts, baseline = (
                self._compute_shap(
                    model,
                    X
                )
            )

            baseline_values.append(
                baseline
            )

            all_shaps.append(
                [
                    impact["impact"]
                    for impact in impacts
                ]
            )

        mean_baseline = float(
            np.mean(
                baseline_values
            )
        )

        mean_impacts = np.mean(
            np.array(all_shaps),
            axis=0
        )

        template = impacts

        ensemble_impacts = []

        for tmpl, impact in zip(
            template,
            mean_impacts
        ):

            ensemble_impacts.append(
                {
                    "feature":
                        tmpl["feature"],
                    "value":
                        tmpl["value"],
                    "impact":
                        round(
                            float(impact),
                            4
                        ),
                    "direction":
                        (
                            "increases_risk"
                            if impact > 0
                            else "decreases_risk"
                        )
                }
            )

        ensemble_impacts = sorted(
            ensemble_impacts,
            key=lambda x: abs(
                x["impact"]
            ),
            reverse=True
        )

        return (
            ensemble_impacts[:3],
            mean_baseline
        )

    def predict(
        self,
        case_id: str,
        structured_prompt: dict
    ) -> dict[str, Any]:
        
        X = self._build_feature_row(
            structured_prompt
        )   

        if case_id in self.case_to_fold:

            fold_id = self.case_to_fold[
                case_id
            ]

            model = self.models[
                fold_id
            ]

            probability = (
                self._predict_single_model(
                    model,
                    X
                )
            )

            (
                shap_explanation,
                baseline_value
            ) = self._compute_shap(
                model,
                X
            )

            shap_explanation = (
                shap_explanation[:3]
            )

            source = "out_of_fold"

        else:

            probabilities = [
                self._predict_single_model(
                    model,
                    X
                )
                for model in self.models
            ]

            probability = float(
                np.mean(probabilities)
            )

            (
                shap_explanation,
                baseline_value
            ) = self._ensemble_shap(
                X
            )

            source = "ensemble"
            
        if probability < 0.20:
            risk = "low"
        elif probability < 0.40:
            risk = "moderate"
        elif probability < 0.70:
            risk = "high"
        else:
            risk = "very_high"
        
        results = {

            "catboost_biopsy_risk_percent":
            round(
                probability * 100,
                1
            ),
            "ml_clinical_risk_category": risk,
                

            "prediction":
                (
                    "recommend_biopsy"
                    if probability >= self.threshold
                    else "avoid_biopsy"
                ),

            "source":
                source,

            "model_auc":
                self.cv_auc,

            "threshold":
                self.threshold,

            "baseline_value":
                round(
                    baseline_value,
                    4
                ),

            "global_feature_importance":
                self.global_feature_importance,

            "top_local_factors":
                shap_explanation,
        }

      
        return results
        

import json

PREDICTOR_TOOL_NAME = "get_biopsy_risk_prediction"


def make_task1_catboost_tool(case_store):
    predictor = Task1CatBoostPredictor()

    def get_biopsy_risk_prediction(case_id: str) -> str:

        case = case_store.get_case(case_id)

        if case is None:
            return json.dumps({
                "error": f"Case not found: {case_id}"
            })

        result = predictor.predict(
            case_id=case_id,
            structured_prompt=case
        )

        result = make_json_safe(result)

        return json.dumps({
            "case_id": case_id,
            **result
        })

    get_biopsy_risk_prediction.__name__ = (
        PREDICTOR_TOOL_NAME
    )

    get_biopsy_risk_prediction.__doc__ = (
    "Provides an independent CatBoost-derived clinical risk assessment based "
    "on structured clinical variables. Returns an ML-derived clinical risk "
    "category and feature-level explanations. This estimate complements, but "
    "is distinct from, the MRI-derived 'cspca' probability available in the "
    "patient record."
)

    get_biopsy_risk_prediction.__annotations__ = {
        "case_id": str,
        "return": str
    }

    return get_biopsy_risk_prediction