"""
Model loading and prediction logic.

Loads the saved sklearn model, scaler, and feature column list once at import
time so they can be reused across requests without reloading from disk.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from io import BytesIO

from feature_extraction import (
    extract_features_from_lines,
    extract_features_from_file,
    get_report_from_lines,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DIAGNOSIS_NAMES = {0: "Control", 1: "MCI"}
MULTIMODAL_LABELS = {
    0: "Mild Impairment",
    1: "Moderate Impairment",
    2: "No Impairment",
    3: "Very Mild Impairment",
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
MULTIMODAL_MODEL_DIR = os.path.join(MODEL_DIR, "multimodal")
MRI_MODEL_PATH = os.environ.get(
    "MRI_MODEL_PATH",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "Alzheimers-Disease-Classification",
            "Somesh_VGG16.h5",
        )
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_scaler = None
_feature_cols = None
_multimodal_model = None
_multimodal_feature_cols = None
_mri_model = None
_cnn_cha_model = None
_cnn_cha_scaler = None
_cnn_cha_feature_cols = None
_cnn_cha_default_threshold = None

CNN_CHA_ARTIFACT_DIR = os.path.abspath(
    os.environ.get(
        "CNN_CHA_ARTIFACT_DIR",
        os.path.join(os.path.dirname(__file__), "model_artifacts", "cnn_cha_fusion"),
    )
)
CNN_CHA_FALLBACK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "cnn_cha_fusion", "artifacts")
)
CNN_CHA_IMG_SIZE = 180


def _resolve_cnn_cha_artifact_file(filename: str, required: bool = True) -> Optional[str]:
    primary = os.path.join(CNN_CHA_ARTIFACT_DIR, filename)
    if os.path.exists(primary):
        return primary

    fallback = os.path.join(CNN_CHA_FALLBACK_DIR, filename)
    if os.path.exists(fallback):
        return fallback

    if required:
        raise FileNotFoundError(
            f"Missing CNN+CHA artifact '{filename}'. "
            f"Expected in '{CNN_CHA_ARTIFACT_DIR}' or fallback '{CNN_CHA_FALLBACK_DIR}'."
        )
    return None


def _sanitize_model_config(obj):
    """Recursively sanitize Keras model config for cross-version compatibility."""
    if isinstance(obj, dict):
        class_name = obj.get("class_name")
        cfg = obj.get("config")
        if class_name == "InputLayer" and isinstance(cfg, dict):
            if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                cfg["batch_input_shape"] = cfg.pop("batch_shape")
            cfg.pop("optional", None)

        for k, v in list(obj.items()):
            obj[k] = _sanitize_model_config(v)
        return obj
    if isinstance(obj, list):
        return [_sanitize_model_config(v) for v in obj]
    return obj


def _load_model_compat(model_path: str):
    """Load a Keras model with fallback for version-mismatch config keys."""
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required to load MRI model artifacts.") from exc

    try:
        return tf.keras.models.load_model(model_path)
    except ValueError as err:
        err_msg = str(err)
        if "batch_shape" not in err_msg and "optional" not in err_msg:
            raise

        import h5py

        with h5py.File(model_path, "r") as f:
            raw = f.attrs.get("model_config")
            if raw is None:
                raise
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

        model_cfg = json.loads(raw)
        model_cfg = _sanitize_model_config(model_cfg)
        repaired_model = tf.keras.models.model_from_json(json.dumps(model_cfg))
        repaired_model.load_weights(model_path)
        return repaired_model


def _candidate_model_paths(model_path: str) -> list[str]:
    """Return preferred fallback MRI model files in the same directory."""
    model_dir = os.path.dirname(model_path)
    preferred = [
        os.path.basename(model_path),
        "VGG16.h5",
        "vgg16_97.h5",
        "vgg16_98.h5",
        "Alzheimers_VGG16_Split.h5",
        "Somesh_VGG16.h5",
    ]

    seen = set()
    candidates = []
    for name in preferred:
        p = os.path.abspath(os.path.join(model_dir, name))
        if p not in seen and os.path.exists(p):
            seen.add(p)
            candidates.append(p)
    return candidates


def _load_artifacts():
    """Load model artefacts from disk (called once on first prediction)."""
    global _model, _scaler, _feature_cols

    model_path = os.path.join(MODEL_DIR, "ad_mci_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    for p in (model_path, scaler_path, features_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing model artifact: {p}. "
                "Run the training notebook (Step 17) first."
            )

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    _feature_cols = joblib.load(features_path)


def get_artifacts():
    """Return (model, scaler, feature_cols), loading from disk if needed."""
    if _model is None:
        _load_artifacts()
    return _model, _scaler, _feature_cols


def _load_multimodal_artifacts():
    global _multimodal_model, _multimodal_feature_cols

    model_path = os.path.join(MULTIMODAL_MODEL_DIR, "multimodal_rf_model.pkl")
    feature_path = os.path.join(MULTIMODAL_MODEL_DIR, "multimodal_feature_names.pkl")

    for p in (model_path, feature_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing multimodal artifact: {p}. "
                "Train with classification/train_multimodal_random_forest.py first."
            )

    _multimodal_model = joblib.load(model_path)
    _multimodal_feature_cols = joblib.load(feature_path)


def get_multimodal_artifacts():
    if _multimodal_model is None:
        _load_multimodal_artifacts()
    return _multimodal_model, _multimodal_feature_cols


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _safe_max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _extract_cnn_cha_features_from_lines(lines: list[str]) -> Optional[dict]:
    silences, par_summary, word_segments, response_times = get_report_from_lines(lines)
    if len(word_segments) == 0:
        return None

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
    }


def _load_cnn_cha_artifacts():
    global _cnn_cha_model, _cnn_cha_scaler, _cnn_cha_feature_cols, _cnn_cha_default_threshold

    model_path = _resolve_cnn_cha_artifact_file("cnn_cha_fusion_v2.keras", required=True)
    scaler_path = _resolve_cnn_cha_artifact_file("voice_scaler.pkl", required=True)
    feature_path = _resolve_cnn_cha_artifact_file("voice_feature_columns.pkl", required=True)
    report_path = _resolve_cnn_cha_artifact_file("cnn_cha_fusion_v2_report.json", required=False)

    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for CNN+CHA multimodal prediction.") from exc

    _cnn_cha_model = tf.keras.models.load_model(model_path)
    _cnn_cha_scaler = joblib.load(scaler_path)
    _cnn_cha_feature_cols = list(joblib.load(feature_path))
    _cnn_cha_default_threshold = 0.5

    if report_path is not None:
        with open(report_path, "r", encoding="utf-8") as rf:
            report_data = json.load(rf)
        _cnn_cha_default_threshold = float(report_data.get("best_threshold", 0.5))


def get_cnn_cha_artifacts():
    if _cnn_cha_model is None:
        _load_cnn_cha_artifacts()
    return _cnn_cha_model, _cnn_cha_scaler, _cnn_cha_feature_cols, _cnn_cha_default_threshold


def _load_mri_model():
    global _mri_model
    if _mri_model is not None:
        return _mri_model

    if not os.path.exists(MRI_MODEL_PATH):
        raise FileNotFoundError(f"MRI model file not found: {MRI_MODEL_PATH}")

    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for MRI image inference in multimodal endpoint."
        ) from exc

    load_errors = []
    for candidate in _candidate_model_paths(MRI_MODEL_PATH):
        try:
            _mri_model = _load_model_compat(candidate)
            return _mri_model
        except Exception as exc:
            load_errors.append((candidate, str(exc)))

    details = "\n".join([f"- {path}: {err}" for path, err in load_errors])
    raise RuntimeError(
        "Failed to load any compatible MRI model. Tried:\n"
        f"{details}\n"
        "Set MRI_MODEL_PATH to a compatible .h5 model file for this runtime."
    )


def extract_mri_probability_features(image_bytes: bytes) -> dict:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for MRI image decoding.") from exc

    model = _load_mri_model()

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    x = np.asarray(image, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    feature_names = [
        "mri_prob_mild_impairment",
        "mri_prob_moderate_impairment",
        "mri_prob_no_impairment",
        "mri_prob_very_mild_impairment",
    ]

    features = {name: float(probs[i]) for i, name in enumerate(feature_names)}
    features["mri_pred_class_idx"] = int(np.argmax(probs))
    return features


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict_from_cha_lines(lines: list[str], filename: str = "uploaded.cha") -> Optional[dict]:
    """
    Full pipeline: parse .cha lines → extract features → scale → predict.

    Returns a dict with prediction results, or None if feature extraction fails.
    """
    model, scaler, feature_cols = get_artifacts()

    features = extract_features_from_lines(lines)
    if features is None:
        return None

    # Build single-row DataFrame in correct column order
    X = pd.DataFrame([features])[feature_cols].astype(float)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    classes = list(getattr(model, "classes_", range(len(probabilities))))
    class_to_prob = {int(c): float(probabilities[i]) for i, c in enumerate(classes)}
    pred_prob = class_to_prob.get(int(prediction), float(np.max(probabilities)))

    result = {
        "filename": filename,
        "predicted_diagnosis": DIAGNOSIS_NAMES.get(int(prediction), str(prediction)),
        "prediction_code": int(prediction),
        "confidence": round(pred_prob * 100, 2),
        "probabilities": {},
        "features": features,
    }

    for idx, label in DIAGNOSIS_NAMES.items():
        if idx in class_to_prob:
            result["probabilities"][label] = round(class_to_prob[idx] * 100, 2)

    return result


def predict_from_cha_file(file_path: str) -> Optional[dict]:
    """Convenience wrapper — reads a .cha file from disk and predicts."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return predict_from_cha_lines(lines, filename=os.path.basename(file_path))


def predict_multimodal_from_inputs(
    cha_lines: list[str],
    mri_image_bytes: bytes,
    cha_filename: str = "uploaded.cha",
    mri_filename: str = "uploaded_mri.jpg",
) -> Optional[dict]:
    """Fuse transcript voice features and MRI probabilities for a 4-class prediction."""
    model, feature_cols = get_multimodal_artifacts()

    voice_features = extract_features_from_lines(cha_lines)
    if voice_features is None:
        return None

    mri_features = extract_mri_probability_features(mri_image_bytes)

    merged = {**voice_features, **mri_features}
    missing_cols = [c for c in feature_cols if c not in merged]
    if missing_cols:
        raise ValueError(
            "Multimodal feature mismatch. Missing fields required by trained model: "
            f"{missing_cols}"
        )

    X = pd.DataFrame([merged])[feature_cols].astype(float)
    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    classes = list(getattr(model, "classes_", range(len(probs))))
    class_to_prob = {int(c): float(probs[i]) for i, c in enumerate(classes)}
    pred_prob = class_to_prob.get(pred, float(np.max(probs)))

    result = {
        "cha_filename": cha_filename,
        "mri_filename": mri_filename,
        "predicted_diagnosis": MULTIMODAL_LABELS.get(pred, str(pred)),
        "prediction_code": pred,
        "confidence": round(pred_prob * 100, 2),
        "probabilities": {},
        "voice_features": voice_features,
        "mri_features": mri_features,
    }

    for cls, prob in class_to_prob.items():
        label = MULTIMODAL_LABELS.get(cls, str(cls))
        result["probabilities"][label] = round(prob * 100, 2)

    return result


def predict_cnn_cha_from_inputs(
    cha_lines: list[str],
    mri_image_bytes: bytes,
    cha_filename: str = "uploaded.cha",
    mri_filename: str = "uploaded_mri.jpg",
    threshold_override: Optional[float] = None,
) -> Optional[dict]:
    model, scaler, feature_cols, default_threshold = get_cnn_cha_artifacts()

    voice_features = _extract_cnn_cha_features_from_lines(cha_lines)
    if voice_features is None:
        return None

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for MRI image decoding.") from exc

    image = Image.open(BytesIO(mri_image_bytes)).convert("L").resize((CNN_CHA_IMG_SIZE, CNN_CHA_IMG_SIZE))
    image_tensor = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = np.expand_dims(image_tensor, axis=-1)
    image_tensor = np.expand_dims(image_tensor, axis=0)

    missing_cols = [c for c in feature_cols if c not in voice_features]
    if missing_cols:
        raise ValueError(
            "CNN+CHA feature mismatch. Missing fields required by trained model: "
            f"{missing_cols}"
        )

    voice_df = pd.DataFrame(
        [{c: float(voice_features[c]) for c in feature_cols}],
        columns=feature_cols,
    )
    voice_scaled = scaler.transform(voice_df).astype(np.float32)

    prob_mci = float(model.predict([image_tensor, voice_scaled], verbose=0).ravel()[0])
    prob_control = float(1.0 - prob_mci)

    used_threshold = float(default_threshold if threshold_override is None else threshold_override)
    prediction = "MCI" if prob_mci >= used_threshold else "Control"

    return {
        "cha_filename": cha_filename,
        "mri_filename": mri_filename,
        "threshold": round(used_threshold, 4),
        "P(Control)": round(prob_control, 4),
        "P(MCI)": round(prob_mci, 4),
        "Prediction": prediction,
        "voice_features": voice_features,
    }
