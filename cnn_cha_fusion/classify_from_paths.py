from __future__ import annotations

import io
import json
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Must be set before importing TensorFlow to suppress native/runtime logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from PIL import Image

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    root = here.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


ROOT = _resolve_project_root()
from pause_cha_word_by_word import get_report  # noqa: E402


# ----------------------------
# Edit these values and run the file.
# ----------------------------
CHA_FILE = r"E:\ML\silero-python\Delaware\MCI\_soumodip\337-1-v12.cha"
MRI_FILE = ROOT / "Alzheimers-Disease-Classification" / "Combined Dataset" / "test" / "No Impairment" / "9 (11).jpg"
ARTIFACT_DIR = ROOT / "cnn_cha_fusion" / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "cnn_cha_fusion_v2.keras"
SCALER_FILE = ARTIFACT_DIR / "voice_scaler.pkl"
VOICE_COLUMNS_FILE = ARTIFACT_DIR / "voice_feature_columns.pkl"
REPORT_FILE = ARTIFACT_DIR / "cnn_cha_fusion_v2_report.json"

# Keep None to use report best_threshold.
THRESHOLD_OVERRIDE: float | None = None

# Should match training setting.
IMG_SIZE = 180

# Set True only if you want verbose parser logs.
SHOW_PARSER_LOGS = False


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _safe_max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def extract_cha_features(cha_file: Path, silent_parser_logs: bool = True) -> dict[str, Any]:
    if silent_parser_logs:
        with redirect_stdout(io.StringIO()):
            silences, par_summary, word_segments, response_times = get_report(str(cha_file))
    else:
        silences, par_summary, word_segments, response_times = get_report(str(cha_file))

    total_words = len(word_segments)
    total_pauses = len(silences)
    total_speech = float(sum(r.get("total_speech_sec", 0.0) for r in par_summary))
    total_pause = float(sum(r.get("total_silence_sec", 0.0) for r in par_summary))
    total_duration = float(sum(r.get("total_duration_sec", 0.0) for r in par_summary))

    word_durations = [
        float(w.get("duration_sec", 0.0))
        for w in word_segments
        if w.get("duration_sec") is not None
    ]
    silence_durations = [
        float(s.get("silence_duration_sec", 0.0))
        for s in silences
        if s.get("silence_duration_sec") is not None
    ]
    response_durations = [
        float(r.get("response_time_sec", 0.0))
        for r in response_times
        if r.get("response_time_sec") is not None
    ]

    speech_rate_wpm = (total_words / total_speech) * 60.0 if total_speech > 0 and total_words > 0 else 0.0

    return {
        "word_count": int(total_words),
        "pause_count": int(total_pauses),
        "total_speech_time": total_speech,
        "total_pause_time": total_pause,
        "total_duration": total_duration,
        "speech_rate_wpm": float(speech_rate_wpm),
        "pause_per_word_ratio": float(total_pauses / total_words) if total_words > 0 else 0.0,
        "pause_per_speech_sec": float(total_pauses / total_speech) if total_speech > 0 else 0.0,
        "mean_word_duration": _safe_mean(word_durations),
        "std_word_duration": _safe_std(word_durations),
        "mean_silence_duration": _safe_mean(silence_durations),
        "std_silence_duration": _safe_std(silence_durations),
        "max_silence_duration": _safe_max(silence_durations),
        "silence_ratio": float(total_pause / total_duration) if total_duration > 0 else 0.0,
        "response_time_count": int(len(response_durations)),
        "response_time_mean": _safe_mean(response_durations),
        "response_time_std": _safe_std(response_durations),
        "response_time_median": _safe_median(response_durations),
        "patient_id": cha_file.stem,
        "cha_file": str(cha_file),
    }


def load_gray_image(image_path: Path, image_size: int) -> np.ndarray:
    arr = np.asarray(
        Image.open(image_path).convert("L").resize((image_size, image_size)),
        dtype=np.float32,
    ) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_model_artifacts(artifact_dir: Path) -> tuple[tf.keras.Model, Any, list[str], float]:
    model_path = artifact_dir / MODEL_FILE.name
    scaler_path = artifact_dir / SCALER_FILE.name
    cols_path = artifact_dir / VOICE_COLUMNS_FILE.name
    report_path = artifact_dir / REPORT_FILE.name

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
    if not cols_path.exists():
        raise FileNotFoundError(f"Feature columns artifact not found: {cols_path}")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    voice_feature_cols = joblib.load(cols_path)

    threshold = 0.5
    if report_path.exists():
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        threshold = float(report_data.get("best_threshold", 0.5))

    return model, scaler, list(voice_feature_cols), threshold


def predict_from_paths(
    cha_path: Path,
    mri_path: Path,
    artifact_dir: Path,
    image_size: int,
    threshold: float | None,
    silent_parser_logs: bool,
) -> dict[str, Any]:
    model, scaler, voice_feature_cols, default_threshold = load_model_artifacts(artifact_dir)

    cha_features = extract_cha_features(cha_path, silent_parser_logs=silent_parser_logs)

    missing_cols = [c for c in voice_feature_cols if c not in cha_features]
    if missing_cols:
        raise KeyError(f"Missing CHA features required by model: {missing_cols}")

    voice_df = pd.DataFrame(
        [{c: float(cha_features[c]) for c in voice_feature_cols}],
        columns=voice_feature_cols,
    )
    voice_scaled = scaler.transform(voice_df).astype(np.float32)

    image_tensor = load_gray_image(mri_path, image_size=image_size)

    prob_mci = float(model.predict([image_tensor, voice_scaled], verbose=0).ravel()[0])
    prob_control = float(1.0 - prob_mci)

    used_threshold = float(default_threshold if threshold is None else threshold)
    pred_label = "MCI" if prob_mci >= used_threshold else "Control"

    return {
        "cha_file": str(cha_path),
        "mri_file": str(mri_path),
        "threshold": used_threshold,
        "prob_control": prob_control,
        "prob_mci": prob_mci,
        "prediction": pred_label,
    }


def main() -> None:
    cha_path = Path(CHA_FILE).resolve()
    mri_path = Path(MRI_FILE).resolve()
    artifact_dir = Path(ARTIFACT_DIR).resolve()

    if not cha_path.exists():
        raise FileNotFoundError(f"CHA file not found: {cha_path}")
    if not mri_path.exists():
        raise FileNotFoundError(f"MRI image not found: {mri_path}")
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifact_dir}")

    result = predict_from_paths(
        cha_path=cha_path,
        mri_path=mri_path,
        artifact_dir=artifact_dir,
        image_size=int(IMG_SIZE),
        threshold=THRESHOLD_OVERRIDE,
        silent_parser_logs=not SHOW_PARSER_LOGS,
    )

    print(f"CHA file: {result['cha_file']}")
    print(f"MRI file: {result['mri_file']}")
    print(f"Threshold: {result['threshold']:.4f}")
    print(f"P(Control): {result['prob_control']:.4f}")
    print(f"P(MCI): {result['prob_mci']:.4f}")
    print(f"Prediction: {result['prediction']}")


if __name__ == "__main__":
    main()
