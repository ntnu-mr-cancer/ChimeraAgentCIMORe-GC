"""
Task 3 CatBoost recurrence predictor.

Returns:
- recurrence event prediction
- estimated months_to_recurrence
- recurrence risk score
- top local SHAP factors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from chimera_agent_baseline.tools.extraction_tools import extract_features

import numpy as np
import pandas as pd

from catboost import (
    CatBoostRegressor,
    Pool,
)

PREDICTOR_TOOL_NAME = (
    "get_recurrence_prediction"
)

# MODEL_DIR = (
#     Path(__file__).resolve().parents[3]
#     / "experiments"
#     / "task3"
#     / "artifacts_task3"
# )

MODEL_DIR = Path("/opt/ml/model/ml/task3/artifacts_task3")


FEATURES = [
    "bx_isup_max",
    "rp_isup",
    "pt_stage_num",
    "margin_positive",
    "node_positive",
    "lvi",
]


###############################################################################
# JSON SAFETY
###############################################################################


def make_json_safe(obj):

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


###############################################################################
# PREDICTOR
###############################################################################


class Task3RecurrencePredictor:

    def __init__(self):

        self.metadata = json.loads(
            (
                MODEL_DIR
                / "metadata.json"
            ).read_text()
        )

        self.case_to_fold = json.loads(
            (
                MODEL_DIR
                / "fold_assignments.json"
            ).read_text()
        )

        self.models = {}

        for fold in range(5):

            model = CatBoostRegressor()

            model.load_model(
                str(
                    MODEL_DIR
                    / f"fold_{fold}.cbm"
                )
            )

            self.models[fold] = model

        self.ensemble_models = list(
            self.models.values()
        )

    ###########################################################################
    # FEATURES
    ###########################################################################

    def _build_feature_row(
    self,
    clinical_data: dict,
    ) -> pd.DataFrame:

        extracted = extract_features(
            clinical_data
        )

        row = {
            feature: extracted.get(feature)
            for feature in FEATURES
        }

        return pd.DataFrame([row])

    ###########################################################################
    # RISK -> MONTHS
    ###########################################################################

    def _risk_to_months(
        self,
        risk_score: float,
    ) -> float:

        risk_min = self.metadata[
            "risk_min"
        ]

        risk_max = self.metadata[
            "risk_max"
        ]

        time_min = self.metadata[
            "time_min"
        ]

        time_max = self.metadata[
            "time_max"
        ]

        #
        # smooth sigmoid mapping
        #
        # high risk
        #   ->
        # low months
        #
        # low risk
        #   ->
        # high months
        #

        mid = (
            risk_min
            + risk_max
        ) / 2

        scale = (
            risk_max
            - risk_min
        ) / 6

        p = 1.0 / (
            1.0
            + np.exp(
                (
                    risk_score
                    - mid
                )
                / scale
            )
        )

        months = (
            time_min
            + p
            * (
                time_max
                - time_min
            )
        )

        return float(months)

    ###########################################################################
    # SHAP
    ###########################################################################

    def _compute_shap(
        self,
        model,
        X: pd.DataFrame,
    ):

        pool = Pool(X)

        shap_raw = (
            model.get_feature_importance(
                pool,
                type="ShapValues",
            )
        )

        shap_values = np.abs(
            shap_raw[0, :-1]
        )

        results = []

        for feature, value, impact in zip(
            FEATURES,
            X.iloc[0].values,
            shap_values,
        ):

            results.append(
                {
                    "feature":
                        feature,

                    "value":
                        (
                            None
                            if pd.isna(value)
                            else value
                        ),

                    "impact":
                        round(
                            float(impact),
                            4,
                        ),
                }
            )

        results = sorted(
            results,
            key=lambda x:
                x["impact"],
            reverse=True,
        )

        return results

    def _ensemble_shap(
        self,
        X: pd.DataFrame,
    ):

        feature_scores = {}

        feature_values = {}

        for model in self.ensemble_models:

            factors = self._compute_shap(
                model,
                X,
            )

            for factor in factors:

                feature = factor[
                    "feature"
                ]

                feature_scores.setdefault(
                    feature,
                    [],
                ).append(
                    factor["impact"]
                )

                feature_values[
                    feature
                ] = factor["value"]

        results = []

        for feature, impacts in (
            feature_scores.items()
        ):

            results.append(
                {
                    "feature":
                        feature,

                    "value":
                        feature_values[
                            feature
                        ],

                    "impact":
                        round(
                            float(
                                np.mean(
                                    impacts
                                )
                            ),
                            4,
                        ),
                }
            )

        results = sorted(
            results,
            key=lambda x:
                x["impact"],
            reverse=True,
        )

        return results

    ###########################################################################
    # PREDICT
    ###########################################################################

    def predict(
        self,
        case_id: str,
        clinical_data: dict,
    ) -> dict[str, Any]:

        X = self._build_feature_row(
            clinical_data
        )

        #######################################################################
        # OOF CASE
        #######################################################################

        if case_id in self.case_to_fold:

            fold = int(
                self.case_to_fold[
                    case_id
                ]
            )

            model = self.models[
                fold
            ]

            risk_score = float(
                model.predict(X)[0]
            )

            shap_factors = (
                self._compute_shap(
                    model,
                    X,
                )
            )

            source = "out_of_fold"

        #######################################################################
        # UNSEEN CASE
        #######################################################################

        else:

            risks = []

            for model in self.ensemble_models:

                risks.append(
                    float(
                        model.predict(X)[0]
                    )
                )

            risk_score = float(
                np.mean(risks)
            )

            shap_factors = (
                self._ensemble_shap(
                    X
                )
            )

            source = "ensemble"

        #######################################################################
        # EVENT
        #######################################################################

        event_threshold = (
            self.metadata[
                "event_threshold"
            ]
        )

        predicted_event = int(
            risk_score
            >= event_threshold
        )

        #######################################################################
        # MONTHS
        #######################################################################

        months_to_recurrence = (
            self._risk_to_months(
                risk_score
            )
        )

        important_factors = [
            factor["feature"]
            for factor in shap_factors[:3]
        ]

        #######################################################################
        # RESULT
        #######################################################################

        return {

            "event":
                predicted_event,

            "months_to_recurrence":
                round(
                    months_to_recurrence,
                    1,
                ),

            "risk_score":
                round(
                    risk_score,
                    4,
                ),

            "important_factors":
                important_factors,

            "top_local_factors":
                shap_factors[:5],

            "source":
                source,

            "model_metadata":
                {
                    "oof_c_index":
                        self.metadata.get(
                            "oof_c_index"
                        ),
                },
        }


###############################################################################
# TOOL WRAPPER
###############################################################################


def make_task3_catboost_tool(
    store,
):

    predictor = (
        Task3RecurrencePredictor()
    )

    def get_recurrence_prediction(
        case_id: str,
    ) -> str:

        case = store.get_case(
            case_id
        )

        if case is None:

            return json.dumps(
                {
                    "error":
                        f"Case not found: {case_id}"
                }
            )

        result = predictor.predict(
            case_id=case_id,
            clinical_data=case,
        )

        result = make_json_safe(
            result
        )

        return json.dumps(
            {
                "case_id":
                    case_id,
                **result,
            }
        )

    get_recurrence_prediction.__name__ = (
        PREDICTOR_TOOL_NAME
    )

    get_recurrence_prediction.__annotations__ = {
        "case_id": str,
        "return": str,
    }

    get_recurrence_prediction.__doc__ = (
        "Provides an independent CatBoost-based biochemical recurrence "
        "prediction following radical prostatectomy. Returns predicted "
        "recurrence event status, estimated months to recurrence, Cox risk "
        "score, and SHAP-based explanations of the most influential "
        "patient-specific pathological risk factors."
    )

    return (
        get_recurrence_prediction
    )