# train_catboost.py

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
)

from data_utils import load_dataset, build_preprocessor

DATA_PATH = "/work/michaesl/prostate/ChimeraAgentCIMORe/data/task1/agent_input"


FEATURES = [
    "pirads",
    "bx",
    "psa",
    "age",
    "psad",
]

CAT_COLS = [
    "pirads",
    "bx",
]

NUM_COLS = [
    "psa",
    "age",
    "psad",
]

MODEL_DIR = Path("artifacts/catboost")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


    
df = load_dataset(DATA_PATH)



X = df[FEATURES].copy()
y = df["target"]

case_ids = df["case_id"].tolist()


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probs = np.zeros(len(df))

case_to_fold = {}

feature_importances = []

models = []



for fold_id, (train_idx, test_idx) in enumerate(
    cv.split(X, y)
):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]

    preprocessor = build_preprocessor(NUM_COLS=NUM_COLS, CAT_COLS=CAT_COLS )

    model = CatBoostClassifier(
        iterations=300,
        depth=2,
        learning_rate=0.03,
        loss_function="Logloss",
        verbose=False,
        random_seed=42
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]

    oof_probs[test_idx] = probs

    for idx in test_idx:

        case_to_fold[
            case_ids[idx]
        ] = fold_id

    importances = (
        pipe.named_steps["model"]
        .get_feature_importance()
    )

    feature_importances.append(importances)

    model_path = (
        MODEL_DIR /
        f"fold_{fold_id}.joblib"
    )

    joblib.dump(
        pipe,
        model_path
    )

    models.append(model_path)

    print(
        f"Fold {fold_id} complete"
    )
    


threshold = 0.5

oof_preds = (
    oof_probs >= threshold
).astype(int)

auc = roc_auc_score(
    y,
    oof_probs
)

accuracy = accuracy_score(
    y,
    oof_preds
)

f1 = f1_score(
    y,
    oof_preds
)


print(f"\nOOF AUC: {auc:.3f}")
print(f"OOF Accuracy: {accuracy:.3f}")
print(f"OOF F1: {f1:.3f}")


from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

auc = roc_auc_score(y, oof_probs)

# --------------------------------------------------
# Threshold search
# --------------------------------------------------

thresholds = np.arange(0.05, 0.96, 0.01)

results = []

for threshold in thresholds:

    preds = (oof_probs >= threshold).astype(int)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
    })

threshold_df = pd.DataFrame(results)

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])
best_accuracy = float(best_row["accuracy"])
best_f1 = float(best_row["f1"])
best_precision = float(best_row["precision"])
best_recall = float(best_row["recall"])

print(f"\nOOF AUC: {auc:.3f}")
print(f"Best threshold: {best_threshold:.2f}")
print(f"Accuracy: {best_accuracy:.3f}")
print(f"F1: {best_f1:.3f}")
print(f"Precision: {best_precision:.3f}")
print(f"Recall: {best_recall:.3f}")


# mean_importance = np.mean(
#     feature_importances,
#     axis=0
# )

# feature_importance_dict = {
#     feature: float(importance)
#     for feature, importance in zip(
#         FEATURES,
#         mean_importance
#     )
# }

# metadata = {
#     "features": FEATURES,
#     "cv_auc": float(auc),
#     "n_models": 5,
#     "feature_importance":
#         feature_importance_dict
# }

# with open(
#     MODEL_DIR / "metadata.json",
#     "w"
# ) as f:

#     json.dump(
#         metadata,
#         f,
#         indent=2
#     )
  
# oof_df = pd.DataFrame({
#     "case_id": case_ids,
#     "target": y,
#     "probability": oof_probs
# })

# oof_df.to_csv(
#     MODEL_DIR / "oof_predictions.csv",
#     index=False
# )  

# importance_df = pd.DataFrame({
#     "feature": FEATURES,
#     "importance": mean_importance
# }).sort_values(
#     "importance",
#     ascending=False
# )

# importance_df.to_csv(
#     MODEL_DIR / "feature_importance.csv",
#     index=False
# )
    
# with open(
#     MODEL_DIR / "case_to_fold.json",
#     "w"
# ) as f:

#     json.dump(
#         case_to_fold,
#         f,
#         indent=2
#     )
    
    
    
# from catboost import Pool

# # --------------------------------------------------
# # Global SHAP analysis
# # --------------------------------------------------

# all_shap = []

# for model_path in models:

#     pipe = joblib.load(model_path)

#     X_proc = pipe.named_steps["prep"].transform(X)

#     shap_values = pipe.named_steps["model"].get_feature_importance(
#         Pool(X_proc),
#         type="ShapValues"
#     )

#     # Last column is bias term
#     shap_values = shap_values[:, :-1]

#     all_shap.append(
#         np.abs(shap_values)
#     )

# all_shap = np.vstack(all_shap)

# mean_abs_shap = all_shap.mean(axis=0)

# shap_df = pd.DataFrame({
#     "feature": FEATURES,
#     "mean_abs_shap": mean_abs_shap
# }).sort_values(
#     "mean_abs_shap",
#     ascending=False
# )

# print("\nMean absolute SHAP values:")
# print(shap_df)

# shap_df.to_csv(
#     MODEL_DIR / "shap_importance.csv",
#     index=False
# )

# from catboost import Pool

# all_abs_shap_yes = []
# all_abs_shap_no = []

# all_signed_shap_yes = []
# all_signed_shap_no = []

# for model_path in models:

#     pipe = joblib.load(model_path)

#     X_proc = pipe.named_steps["prep"].transform(X)

#     shap_values = pipe.named_steps["model"].get_feature_importance(
#         Pool(X_proc),
#         type="ShapValues",
#     )

#     # Remove bias column
#     shap_values = shap_values[:, :-1]

#     abs_shap = np.abs(shap_values)

#     # Absolute SHAP
#     all_abs_shap_yes.append(
#         abs_shap[y.values == 1]
#     )

#     all_abs_shap_no.append(
#         abs_shap[y.values == 0]
#     )

#     # Signed SHAP
#     all_signed_shap_yes.append(
#         shap_values[y.values == 1]
#     )

#     all_signed_shap_no.append(
#         shap_values[y.values == 0]
#     )


# # --------------------------------------------------
# # Stack
# # --------------------------------------------------

# all_abs_shap_yes = np.vstack(all_abs_shap_yes)
# all_abs_shap_no = np.vstack(all_abs_shap_no)

# all_signed_shap_yes = np.vstack(all_signed_shap_yes)
# all_signed_shap_no = np.vstack(all_signed_shap_no)


# # --------------------------------------------------
# # Absolute SHAP
# # --------------------------------------------------

# mean_abs_yes = all_abs_shap_yes.mean(axis=0)
# mean_abs_no = all_abs_shap_no.mean(axis=0)

# abs_yes_df = (
#     pd.DataFrame({
#         "feature": FEATURES,
#         "mean_abs_shap_yes": mean_abs_yes,
#     })
#     .sort_values("mean_abs_shap_yes", ascending=False)
# )

# abs_no_df = (
#     pd.DataFrame({
#         "feature": FEATURES,
#         "mean_abs_shap_no": mean_abs_no,
#     })
#     .sort_values("mean_abs_shap_no", ascending=False)
# )

# print("\n=== ABSOLUTE SHAP FOR BIOPSY YES ===")
# print(abs_yes_df)

# print("\n=== ABSOLUTE SHAP FOR BIOPSY NO ===")
# print(abs_no_df)


# # --------------------------------------------------
# # Signed SHAP
# # --------------------------------------------------

# mean_signed_yes = all_signed_shap_yes.mean(axis=0)
# mean_signed_no = all_signed_shap_no.mean(axis=0)

# signed_df = pd.DataFrame({
#     "feature": FEATURES,
#     "mean_signed_yes": mean_signed_yes,
#     "mean_signed_no": mean_signed_no,
# })

# signed_df["difference"] = (
#     signed_df["mean_signed_yes"]
#     - signed_df["mean_signed_no"]
# )

# signed_df = signed_df.sort_values(
#     "difference",
#     ascending=False,
# )

# print("\n=== SIGNED SHAP COMPARISON ===")
# print(signed_df)


# # --------------------------------------------------
# # Save
# # --------------------------------------------------

# abs_yes_df.to_csv(
#     MODEL_DIR / "shap_importance_yes.csv",
#     index=False,
# )

# abs_no_df.to_csv(
#     MODEL_DIR / "shap_importance_no.csv",
#     index=False,
# )

# signed_df.to_csv(
#     MODEL_DIR / "shap_signed_comparison.csv",
#     index=False,
# )
