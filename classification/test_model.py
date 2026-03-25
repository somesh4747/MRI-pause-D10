"""
Test saved model on new patients.

Usage:
    # Single .cha file
    python test_model.py --cha "E:\ML\silero-python\Delaware\MCI\01-1.cha"

    # All .cha files in a directory
    python test_model.py --dir "E:\ML\silero-python\Delaware\MCI"

    # Pre-built CSV (must contain the same feature columns used during training)
    python test_model.py --csv "E:\ML\silero-python\training_dataset_TESTING.csv"

    # Optional: specify model artifacts directory (default: ../model_artifacts)
    python test_model.py --cha "..." --model-dir "E:\ML\silero-python\model_artifacts"
"""

import sys
import os
import argparse
import glob

import pandas as pd
import numpy as np
import joblib

# Add parent directory to path so we can import _main_features
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _main_features import extract_features_from_patient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DIAGNOSIS_NAMES = {0: "Control", 1: "MCI"}

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")


def load_model_artifacts(model_dir: str):
    """Load the saved model, scaler, and feature column names."""
    model_dir = os.path.abspath(model_dir)

    model_path = os.path.join(model_dir, "ad_mci_model.pkl")
    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    features_path = os.path.join(model_dir, "feature_names.pkl")

    for p in (model_path, scaler_path, features_path):
        if not os.path.exists(p):
            print(f"ERROR: Missing artifact — {p}")
            print("Run the training notebook first (Step 17) to save model artifacts.")
            sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)

    print(f"Model loaded from: {model_dir}")
    print(f"  Model type : {type(model).__name__}")
    print(f"  Features   : {feature_cols}")
    return model, scaler, feature_cols


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_single_cha(cha_file: str, model, scaler, feature_cols):
    """Extract features from a .cha file, scale, and predict."""
    print(f"\nExtracting features from: {cha_file}")
    features = extract_features_from_patient(cha_file)

    if features is None:
        print("  ✗ Could not extract features — skipping.")
        return None

    # Build a single-row DataFrame in the correct column order
    X_new = pd.DataFrame([features])[feature_cols].astype(float)
    X_scaled = scaler.transform(X_new)

    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    result = {
        "patient_file": os.path.basename(cha_file),
        "predicted_diagnosis": DIAGNOSIS_NAMES.get(prediction, str(prediction)),
        "prediction_code": int(prediction),
        "confidence": round(float(probabilities[prediction]) * 100, 2),
    }

    # Add per-class probabilities
    for idx, col_name in DIAGNOSIS_NAMES.items():
        if idx < len(probabilities):
            result[f"prob_{col_name}"] = round(float(probabilities[idx]) * 100, 2)

    return result


def predict_from_csv(csv_path: str, model, scaler, feature_cols):
    """Load a CSV that already has feature columns, scale, and predict."""
    df = pd.read_csv(csv_path)
    print(f"\nLoaded CSV: {csv_path}  ({len(df)} rows)")

    # Check that all required feature columns are present
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"ERROR: CSV is missing these feature columns: {missing}")
        sys.exit(1)

    X = df[feature_cols].astype(float)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    # Build results
    results = []
    for i in range(len(df)):
        pred = predictions[i]
        probs = probabilities[i]

        row = {
            "patient_id": df["patient_id"].iloc[i] if "patient_id" in df.columns else i,
            "predicted_diagnosis": DIAGNOSIS_NAMES.get(pred, str(pred)),
            "prediction_code": int(pred),
            "confidence": round(float(probs[pred]) * 100, 2),
        }

        for idx, col_name in DIAGNOSIS_NAMES.items():
            if idx < len(probs):
                row[f"prob_{col_name}"] = round(float(probs[idx]) * 100, 2)

        # If true labels exist, include them for comparison
        if "diagnosis" in df.columns:
            true_label = int(df["diagnosis"].iloc[i])
            row["true_diagnosis"] = DIAGNOSIS_NAMES.get(true_label, str(true_label))
            row["correct"] = pred == true_label

        results.append(row)

    return results, df


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_single_result(result: dict):
    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print(f"  File       : {result['patient_file']}")
    print(f"  Diagnosis  : {result['predicted_diagnosis']}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  Probabilities:")
    for key in result:
        if key.startswith("prob_"):
            print(f"    {key.replace('prob_', ''):>10}: {result[key]}%")
    print("=" * 60)


def print_batch_summary(results: list):
    print("\n" + "=" * 80)
    print("  BATCH PREDICTION RESULTS")
    print("=" * 80)

    df_res = pd.DataFrame(results)

    # Print each row
    id_col = "patient_id" if "patient_id" in df_res.columns else "patient_file"
    for _, row in df_res.iterrows():
        tag = ""
        if "correct" in row:
            tag = "  ✓" if row["correct"] else "  ✗"
        print(
            f"  {str(row[id_col]):>20}  →  {row['predicted_diagnosis']:<10} "
            f"({row['confidence']:5.1f}%){tag}"
        )

    # Accuracy if true labels available
    if "correct" in df_res.columns:
        acc = df_res["correct"].mean()
        print(f"\n  Accuracy: {acc:.2%}  ({df_res['correct'].sum()}/{len(df_res)})")

    print(f"\n  Diagnosis distribution (predicted):")
    print(f"  {df_res['predicted_diagnosis'].value_counts().to_dict()}")
    print("=" * 80)

    return df_res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test the trained Control/MCI classification model on new data."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cha", type=str, help="Path to a single .cha file")
    group.add_argument("--dir", type=str, help="Directory containing .cha files")
    group.add_argument("--csv", type=str, help="CSV file with pre-computed features")

    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory with model artifacts (default: ../model_artifacts)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save predictions to this CSV file",
    )

    args = parser.parse_args()

    # Load model
    model, scaler, feature_cols = load_model_artifacts(args.model_dir)

    results = []

    # ── Single .cha file ──────────────────────────────────────────────
    if args.cha:
        result = predict_single_cha(args.cha, model, scaler, feature_cols)
        if result:
            print_single_result(result)
            results.append(result)

    # ── Directory of .cha files ───────────────────────────────────────
    elif args.dir:
        cha_files = sorted(glob.glob(os.path.join(args.dir, "*.cha")))
        if not cha_files:
            print(f"No .cha files found in {args.dir}")
            sys.exit(1)

        print(f"\nFound {len(cha_files)} .cha files in {args.dir}\n")
        for cha_file in cha_files:
            result = predict_single_cha(cha_file, model, scaler, feature_cols)
            if result:
                results.append(result)

        if results:
            print_batch_summary(results)

    # ── CSV with pre-computed features ────────────────────────────────
    elif args.csv:
        results, _ = predict_from_csv(args.csv, model, scaler, feature_cols)
        if results:
            print_batch_summary(results)

    # ── Save output ───────────────────────────────────────────────────
    if args.output and results:
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
