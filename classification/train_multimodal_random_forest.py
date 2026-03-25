"""Train an early-fusion Random Forest on voice + MRI features."""

from __future__ import annotations

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


META_COLS = {
    "patient_id",
    "diagnosis",
    "diagnosis_name",
    "canonical_patient_id",
    "target_mri_class",
    "mri_image_path",
    "mri_source_class",
    "pairing_strategy",
    "split",
}


def train_multimodal_model(dataset_csv: str, output_dir: str, seed: int = 42):
    df = pd.read_csv(dataset_csv)

    if "target_4class" not in df.columns:
        raise ValueError("dataset must include target_4class column")
    if "split" not in df.columns:
        raise ValueError("dataset must include split column")
    if "canonical_patient_id" not in df.columns:
        raise ValueError("dataset must include canonical_patient_id column")

    feature_cols = [
        c for c in df.columns if c not in META_COLS and c != "target_4class"
    ]

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["target_4class"].astype(int)
    g_train = train_df["canonical_patient_id"].astype(str)

    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["target_4class"].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_metrics = cross_validate(
        model,
        X_train,
        y_train,
        groups=g_train,
        cv=cv,
        scoring={
            "acc": "accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
            "bal_acc": "balanced_accuracy",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "multimodal_rf_model.pkl")
    features_path = os.path.join(output_dir, "multimodal_feature_names.pkl")
    meta_path = os.path.join(output_dir, "multimodal_training_report.json")

    joblib.dump(model, model_path)
    joblib.dump(feature_cols, features_path)

    summary = {
        "dataset_csv": dataset_csv,
        "num_features": len(feature_cols),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cv_accuracy_mean": float(cv_metrics["test_acc"].mean()),
        "cv_f1_macro_mean": float(cv_metrics["test_f1_macro"].mean()),
        "cv_f1_weighted_mean": float(cv_metrics["test_f1_weighted"].mean()),
        "cv_balanced_accuracy_mean": float(cv_metrics["test_bal_acc"].mean()),
        "test_classification_report": report,
        "test_confusion_matrix": cm,
        "feature_columns": feature_cols,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("MULTIMODAL RANDOM FOREST TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Features: {features_path}")
    print(f"Report: {meta_path}")
    print(f"CV accuracy mean: {summary['cv_accuracy_mean']:.4f}")
    print(f"Test weighted F1: {report['weighted avg']['f1-score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal RF model.")
    parser.add_argument(
        "--dataset-csv",
        default="classification/multimodal_training_dataset.csv",
        help="Input multimodal dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="model_artifacts/multimodal",
        help="Directory to save artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_multimodal_model(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        seed=args.seed,
    )
