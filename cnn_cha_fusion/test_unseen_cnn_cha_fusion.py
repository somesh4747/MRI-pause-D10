from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pause_cha_word_by_word import get_report


def _safe_mean(v):
    return float(np.mean(v)) if len(v) else 0.0


def _safe_std(v):
    return float(np.std(v)) if len(v) else 0.0


def _safe_max(v):
    return float(np.max(v)) if len(v) else 0.0


def _safe_median(v):
    return float(np.median(v)) if len(v) else 0.0


def extract_cha_features(cha_file: Path) -> dict:
    silences, par_summary, word_segments, response_times = get_report(str(cha_file))
    total_words = len(word_segments)
    total_pauses = len(silences)
    total_speech = float(sum(r.get("total_speech_sec", 0.0) for r in par_summary))
    total_pause = float(sum(r.get("total_silence_sec", 0.0) for r in par_summary))
    total_dur = float(sum(r.get("total_duration_sec", 0.0) for r in par_summary))

    word_d = [float(w.get("duration_sec", 0.0)) for w in word_segments if w.get("duration_sec") is not None]
    sil_d = [float(s.get("silence_duration_sec", 0.0)) for s in silences if s.get("silence_duration_sec") is not None]
    resp = [float(r.get("response_time_sec", 0.0)) for r in response_times if r.get("response_time_sec") is not None]

    speech_rate = (total_words / total_speech) * 60.0 if total_speech > 0 and total_words > 0 else 0.0

    return {
        "word_count": int(total_words),
        "pause_count": int(total_pauses),
        "total_speech_time": total_speech,
        "total_pause_time": total_pause,
        "total_duration": total_dur,
        "speech_rate_wpm": float(speech_rate),
        "pause_per_word_ratio": float(total_pauses / total_words) if total_words > 0 else 0.0,
        "pause_per_speech_sec": float(total_pauses / total_speech) if total_speech > 0 else 0.0,
        "mean_word_duration": _safe_mean(word_d),
        "std_word_duration": _safe_std(word_d),
        "mean_silence_duration": _safe_mean(sil_d),
        "std_silence_duration": _safe_std(sil_d),
        "max_silence_duration": _safe_max(sil_d),
        "silence_ratio": float(total_pause / total_dur) if total_dur > 0 else 0.0,
        "response_time_count": int(len(resp)),
        "response_time_mean": _safe_mean(resp),
        "response_time_std": _safe_std(resp),
        "response_time_median": _safe_median(resp),
    }


def load_gray(path: Path, size: int) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=-1)


def find_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in paths]}")


def load_artifacts(artifacts_dir: Path):
    model_path = find_existing([
        artifacts_dir / "cnn_cha_fusion_v2.keras",
        artifacts_dir / "cnn_cha_fusion_final.keras",
    ])

    report_path = find_existing([
        artifacts_dir / "cnn_cha_fusion_v2_report.json",
        artifacts_dir / "cnn_cha_fusion_report.json",
    ])

    scaler_path = find_existing([
        artifacts_dir / "voice_scaler_v2.pkl",
        artifacts_dir / "voice_scaler.pkl",
    ])

    feature_cols_path = find_existing([
        artifacts_dir / "voice_feature_columns_v2.pkl",
        artifacts_dir / "voice_feature_columns.pkl",
        artifacts_dir / "voice_feature_columns_v2.json",
        artifacts_dir / "voice_feature_columns.json",
    ])

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    if feature_cols_path.suffix.lower() == ".json":
        feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    else:
        feature_cols = joblib.load(feature_cols_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    threshold = float(report.get("best_threshold", 0.5))

    return model, scaler, list(feature_cols), threshold, model_path, report_path


def predict_one(
    model,
    scaler,
    feature_cols: list[str],
    cha_file: Path,
    mri_file: Path,
    img_size: int,
    threshold: float,
):
    cha_feats = extract_cha_features(cha_file)
    voice_vec = np.array([[float(cha_feats[c]) for c in feature_cols]], dtype=np.float32)
    voice_scaled = scaler.transform(voice_vec).astype(np.float32)

    img = load_gray(mri_file, img_size)
    img = np.expand_dims(img, axis=0)

    prob_mci = float(model.predict([img, voice_scaled], verbose=0).ravel()[0])
    prob_control = float(1.0 - prob_mci)
    pred = int(prob_mci >= threshold)
    pred_label = "MCI" if pred == 1 else "Control"

    return {
        "cha_file": str(cha_file),
        "mri_file": str(mri_file),
        "prob_control": prob_control,
        "prob_mci": prob_mci,
        "threshold": float(threshold),
        "pred": pred,
        "pred_label": pred_label,
    }


def run_single(args):
    artifacts_dir = Path(args.artifacts_dir)
    model, scaler, feature_cols, threshold, model_path, report_path = load_artifacts(artifacts_dir)

    th = float(args.threshold) if args.threshold is not None else threshold
    out = predict_one(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        cha_file=Path(args.cha_file),
        mri_file=Path(args.mri_file),
        img_size=int(args.img_size),
        threshold=th,
    )

    print(f"MODEL={model_path}")
    print(f"REPORT={report_path}")
    print(json.dumps(out, indent=2))


def _list_files(folder: Path, exts: set[str]) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def _resolve_class_dir(explicit: str | None, root: str | None, candidates: list[str], label: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"{label} path not found: {p}")
        return p

    if not root:
        raise ValueError(f"Missing {label}: provide explicit path or root path.")

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root path not found for {label}: {root_path}")

    for name in candidates:
        p = root_path / name
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not resolve {label} under {root_path}. Tried: {candidates}"
    )


def run_eval(args):
    artifacts_dir = Path(args.artifacts_dir)
    model, scaler, feature_cols, threshold, model_path, report_path = load_artifacts(artifacts_dir)

    th = float(args.threshold) if args.threshold is not None else threshold
    rnd = random.Random(args.seed)

    control_cha_dir = _resolve_class_dir(
        explicit=args.control_cha_dir,
        root=args.cha_root,
        candidates=["Control", "No Impairment"],
        label="Control CHA directory",
    )
    mci_cha_dir = _resolve_class_dir(
        explicit=args.mci_cha_dir,
        root=args.cha_root,
        candidates=["MCI", "Mild Impairment"],
        label="MCI CHA directory",
    )
    control_mri_dir = _resolve_class_dir(
        explicit=args.control_mri_dir,
        root=args.mri_test_root,
        candidates=["No Impairment", "Control"],
        label="Control MRI directory",
    )
    mci_mri_dir = _resolve_class_dir(
        explicit=args.mci_mri_dir,
        root=args.mri_test_root,
        candidates=["Mild Impairment", "MCI"],
        label="MCI MRI directory",
    )

    cha_control = _list_files(control_cha_dir, {".cha"})
    cha_mci = _list_files(mci_cha_dir, {".cha"})
    mri_control = _list_files(control_mri_dir, {".jpg", ".jpeg", ".png"})
    mri_mci = _list_files(mci_mri_dir, {".jpg", ".jpeg", ".png"})

    if not cha_control or not cha_mci or not mri_control or not mri_mci:
        raise RuntimeError("Missing files in one or more input folders for evaluation.")

    def sample_pairs(cha_list: list[Path], mri_list: list[Path], true_label: int, n: int) -> list[dict]:
        rows = []
        for i in range(n):
            cha_file = cha_list[i % len(cha_list)]
            mri_file = mri_list[i % len(mri_list)]
            out = predict_one(
                model=model,
                scaler=scaler,
                feature_cols=feature_cols,
                cha_file=cha_file,
                mri_file=mri_file,
                img_size=int(args.img_size),
                threshold=th,
            )
            out["true_label"] = true_label
            out["true_name"] = "Control" if true_label == 0 else "MCI"
            rows.append(out)
        rnd.shuffle(rows)
        return rows

    rows = []
    if args.mci_only:
        rows.extend(sample_pairs(cha_mci, mri_mci, true_label=1, n=int(args.mci_pairs)))
    else:
        rows.extend(sample_pairs(cha_control, mri_control, true_label=0, n=int(args.pairs_per_class)))
        rows.extend(sample_pairs(cha_mci, mri_mci, true_label=1, n=int(args.pairs_per_class)))

    df = pd.DataFrame(rows)
    y_true = df["true_label"].astype(int).to_numpy()
    y_pred = df["pred"].astype(int).to_numpy()

    print(f"MODEL={model_path}")
    print(f"REPORT={report_path}")
    print(f"USED_THRESHOLD={th:.4f}")
    print(f"CONTROL_CHA_DIR={control_cha_dir}")
    print(f"MCI_CHA_DIR={mci_cha_dir}")
    print(f"CONTROL_MRI_DIR={control_mri_dir}")
    print(f"MCI_MRI_DIR={mci_mri_dir}")

    if args.mci_only:
        # In MCI-only evaluation, report MCI recall directly.
        total = int(len(df))
        predicted_mci = int(np.sum(y_pred == 1))
        mci_recall = float(predicted_mci / total) if total > 0 else 0.0
        print("MCI_ONLY_SUMMARY")
        print(f"TOTAL_MCI_PAIRS={total}")
        print(f"PREDICTED_MCI={predicted_mci}")
        print(f"MCI_RECALL={mci_recall:.4f}")
    else:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        print("CONFUSION_MATRIX")
        print(cm)
        print("CLASSIFICATION_REPORT")
        print(classification_report(y_true, y_pred, target_names=["Control", "MCI"], digits=4, zero_division=0))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"OUT_CSV={out_csv}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run unseen-data inference for CNN+CHA fusion model.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pred = sub.add_parser("predict", help="Predict for one CHA + one MRI pair.")
    p_pred.add_argument("--cha-file", required=True)
    p_pred.add_argument("--mri-file", required=True)
    p_pred.add_argument("--artifacts-dir", default=str(ROOT / "cnn_cha_fusion" / "artifacts"))
    p_pred.add_argument("--img-size", type=int, default=180)
    p_pred.add_argument("--threshold", type=float, default=None)

    p_eval = sub.add_parser("eval", help="Evaluate on unseen folders with known mapping.")
    p_eval.add_argument("--cha-root", default=str(ROOT / "Delaware"), help="Root containing CHA class folders (e.g., Control, MCI).")
    p_eval.add_argument("--mri-test-root", default=str(ROOT / "Alzheimers-Disease-Classification" / "Combined Dataset" / "test"), help="MRI test root containing class folders (e.g., No Impairment, Mild Impairment).")
    p_eval.add_argument("--control-cha-dir", default=None, help="Optional explicit Control CHA folder. Overrides --cha-root.")
    p_eval.add_argument("--mci-cha-dir", default=None, help="Optional explicit MCI CHA folder. Overrides --cha-root.")
    p_eval.add_argument("--control-mri-dir", default=None, help="Optional explicit Control MRI folder. Overrides --mri-test-root.")
    p_eval.add_argument("--mci-mri-dir", default=None, help="Optional explicit MCI MRI folder. Overrides --mri-test-root.")
    p_eval.add_argument("--pairs-per-class", type=int, default=150)
    p_eval.add_argument("--mci-only", action="store_true", help="Evaluate only MCI pairs.")
    p_eval.add_argument("--mci-pairs", type=int, default=300, help="Number of MCI pairs when --mci-only is used.")
    p_eval.add_argument("--artifacts-dir", default=str(ROOT / "cnn_cha_fusion" / "artifacts"))
    p_eval.add_argument("--img-size", type=int, default=180)
    p_eval.add_argument("--threshold", type=float, default=None)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--output-csv", default=str(ROOT / "cnn_cha_fusion" / "data" / "unseen_eval_predictions.csv"))

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "predict":
        run_single(args)
    elif args.command == "eval":
        run_eval(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
