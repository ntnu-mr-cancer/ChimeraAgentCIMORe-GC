"""
Task 2 CatBoost predictor.

Returns:
- treatment prediction
- treatment probabilities
- confidence estimate
- variable weight estimates
- top local SHAP factors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catboost import (
    CatBoostClassifier,
    Pool,
)

PREDICTOR_TOOL_NAME = (
    "get_treatment_recommendation_prediction"
)

# MODEL_DIR = (
#     Path(__file__).resolve().parents[3]
#     / "experiments"
#     / "task2"
#     / "artifacts_task2"
# )
MODEL_DIR = Path("/opt/ml/model/ml/task2/artifacts_task2")




FEATURES = [
    "bx_isup",
    "pirads",
    "psa",
    "age",
    "ct",
    "bx_gl_prim",
    "bx_gl_sec",
    "comorbidity",
    "psad",
]

CAT_COLS = [
    "bx_isup",
    "pirads",
    "ct",
]


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


class Task2TreatmentPredictor:

    def __init__(self):

        self.case_to_fold = json.loads(
            (
                MODEL_DIR
                / "fold_assignments.json"
            ).read_text()
        )

        self.metadata = json.loads(
            (
                MODEL_DIR
                / "metadata.json"
            ).read_text()
        )

        self.models = {}
        self.weight_priors = {}
        self.confidence_priors = {}

        n_splits = self.metadata[
            "n_splits"
        ]

        for fold in range(
            n_splits
        ):

            model = CatBoostClassifier()

            model.load_model(
                str(
                    MODEL_DIR
                    / f"treatment_model_fold_{fold}.cbm"
                )
            )

            self.models[fold] = model

            self.weight_priors[
                fold
            ] = json.loads(
                (
                    MODEL_DIR
                    / f"weight_priors_fold_{fold}.json"
                ).read_text()
            )

            self.confidence_priors[
                fold
            ] = json.loads(
                (
                    MODEL_DIR
                    / f"confidence_priors_fold_{fold}.json"
                ).read_text()
            )

        self.ensemble_models = list(
            self.models.values()
        )

    ###########################################################################
    # FEATURES
    ###########################################################################

    def _build_feature_row(
        self,
        structured_prompt: dict,
    ) -> pd.DataFrame:

        row = {}

        for feature in FEATURES:
            row[feature] = (
                structured_prompt.get(
                    feature
                )
            )

        X = pd.DataFrame([row])

        for col in CAT_COLS:

            X[col] = (
                X[col]
                .fillna("missing")
                .astype(str)
            )

        return X

    ###########################################################################
    # SHAP
    ###########################################################################

    def _compute_shap(
        self,
        model,
        X: pd.DataFrame,
    ):

        pool = Pool(
            X,
            cat_features=CAT_COLS,
        )

        shap_raw = (
            model.get_feature_importance(
                pool,
                type="ShapValues",
            )
        )

        #
        # multiclass:
        # [n_samples, n_classes, n_features+1]
        #

        if len(shap_raw.shape) == 3:

            shap_values = (
                np.abs(
                    shap_raw[:, :, :-1]
                )
                .mean(axis=1)
            )[0]

        else:

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
    # PRIORS
    ###########################################################################

    def _predict_confidence(
        self,
        fold,
        treatment,
    ):

        probs = (
            self
            .confidence_priors
            .get(fold, {})
            .get(treatment, {})
        )

        if not probs:
            return "clear"

        return max(
            probs.items(),
            key=lambda x: x[1],
        )[0]

    def _predict_variable_weights(
        self,
        fold,
        treatment,
    ):

        priors = (
            self
            .weight_priors
            .get(fold, {})
            .get(treatment, {})
        )

        weights = {}

        for feature in FEATURES:

            probs = priors.get(
                feature,
                {},
            )

            if probs:

                weights[
                    feature
                ] = max(
                    probs.items(),
                    key=lambda x: x[1],
                )[0]

            else:

                weights[
                    feature
                ] = "not_used"

        return weights

    ###########################################################################
    # PREDICT
    ###########################################################################

    def predict(
        self,
        case_id: str,
        structured_prompt: dict,
    ) -> dict[str, Any]:

        X = self._build_feature_row(
            structured_prompt
        )

        #######################################################################
        # OOF CASE
        #######################################################################

        if case_id in self.case_to_fold:

            fold = self.case_to_fold[
                case_id
            ]

            model = self.models[
                fold
            ]

            probs = (
                model.predict_proba(
                    X
                )[0]
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

            all_probs = []

            for model in self.ensemble_models:

                all_probs.append(
                    model.predict_proba(
                        X
                    )[0]
                )

            probs = np.mean(
                all_probs,
                axis=0,
            )

            shap_factors = (
                self._ensemble_shap(
                    X
                )
            )

            #
            # use fold 0 priors as
            # representative defaults
            #

            fold = 0

            source = "ensemble"

        #######################################################################
        # PREDICTION
        #######################################################################

        classes = list(
            self.models[0].classes_
        )

        probabilities = {

            cls: round(
                float(prob),
                4,
            )

            for cls, prob in zip(
                classes,
                probs,
            )
        }

        predicted_treatment = (
            classes[
                int(
                    np.argmax(
                        probs
                    )
                )
            ]
        )

        confidence = (
            self._predict_confidence(
                fold,
                predicted_treatment,
            )
        )

        variable_weights = (
            self
            ._predict_variable_weights(
                fold,
                predicted_treatment,
            )
        )

        important_factors = [
            factor["feature"]
            for factor in shap_factors[:3]
        ]

        return {

            "prediction":
                predicted_treatment,

            "treatment_probabilities":
                probabilities,

            "predicted_confidence":
                confidence,

            "predicted_variable_weights":
                variable_weights,

            "important_factors":
                important_factors,

            "top_local_factors":
                shap_factors[:5],

            "source":
                source,

            "model_metadata":
                {
                    "oof_macro_f1":
                        self.metadata.get(
                            "oof_macro_f1"
                        ),

                    "oof_weighted_f1":
                        self.metadata.get(
                            "oof_weighted_f1"
                        ),

                    "oof_confidence_score":
                        self.metadata.get(
                            "oof_confidence_score"
                        ),

                    "oof_variable_weight_score":
                        self.metadata.get(
                            "oof_variable_weight_score"
                        ),

                    "oof_factor_score":
                        self.metadata.get(
                            "oof_factor_score"
                        ),

                    "estimated_ranking_score":
                        self.metadata.get(
                            "estimated_ranking_score"
                        ),
                },
        }


###############################################################################
# TOOL WRAPPER
###############################################################################

def make_task2_catboost_tool(
    case_store,
):

    predictor = (
        Task2TreatmentPredictor()
    )

    def get_treatment_recommendation_prediction(
        case_id: str,
    ) -> str:

        case = case_store.get_case(
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
            structured_prompt=case,
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

    get_treatment_recommendation_prediction.__name__ = (
        PREDICTOR_TOOL_NAME
    )

    get_treatment_recommendation_prediction.__annotations__ = {
        "case_id": str,
        "return": str,
    }
    
    get_treatment_recommendation_prediction.__doc__ = (
        "Provides an independent CatBoost-based treatment recommendation for "
        "localized prostate cancer using structured clinical variables. Returns "
        "predicted treatment probabilities, confidence estimation, expected "
        "variable importance labels, and SHAP-based explanations of the most "
        "influential patient-specific factors. Useful for validating and "
        "calibrating treatment decisions against a model trained on historical "
        "expert recommendations."
    )

    return (
        get_treatment_recommendation_prediction
    )