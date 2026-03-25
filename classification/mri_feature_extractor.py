"""Utilities to extract MRI probability features for multimodal fusion."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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
    from tensorflow.keras.models import load_model, model_from_json

    try:
        return load_model(model_path)
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
        repaired_model = model_from_json(json.dumps(model_cfg))
        repaired_model.load_weights(model_path)
        return repaired_model


def _candidate_model_paths(model_path: str) -> list[str]:
    """Return preferred fallback model files in the same directory."""
    model_dir = Path(model_path).parent
    preferred = [
        Path(model_path).name,
        "VGG16.h5",
        "vgg16_97.h5",
        "vgg16_98.h5",
        "Alzheimers_VGG16_Split.h5",
        "Somesh_VGG16.h5",
    ]

    seen = set()
    candidates: list[str] = []
    for name in preferred:
        p = str((model_dir / name).resolve())
        if p not in seen and os.path.exists(p):
            seen.add(p)
            candidates.append(p)
    return candidates


DEFAULT_CLASS_LABELS = {
    0: "Mild Impairment",
    1: "Moderate Impairment",
    2: "No Impairment",
    3: "Very Mild Impairment",
}


@dataclass
class MriFeatureExtractor:
    model_path: str
    image_size: tuple[int, int] = (224, 224)
    class_labels: dict[int, str] | None = None

    def __post_init__(self) -> None:
        self.class_labels = self.class_labels or DEFAULT_CLASS_LABELS
        self._model = None

    def _ensure_model_loaded(self):
        if self._model is not None:
            return self._model

        try:
            from tensorflow.keras.models import load_model  # noqa: F401
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "TensorFlow is required for MRI feature extraction. "
                "Install tensorflow==2.10.0 for this project setup."
            ) from exc

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MRI model file not found: {self.model_path}")

        load_errors = []
        for candidate in _candidate_model_paths(self.model_path):
            try:
                self._model = _load_model_compat(candidate)
                self.model_path = candidate
                return self._model
            except Exception as exc:
                load_errors.append((candidate, str(exc)))

        details = "\n".join([f"- {path}: {err}" for path, err in load_errors])
        raise RuntimeError(
            "Failed to load any compatible MRI model. Tried:\n"
            f"{details}\n"
            "Use a TensorFlow/Keras-compatible .h5 model file or retrain/save with the active runtime."
        )

    def _load_preprocess_image(self, image_path: str) -> np.ndarray:
        try:
            from tensorflow.keras.preprocessing.image import img_to_array, load_img
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "TensorFlow is required for MRI image preprocessing."
            ) from exc

        image = load_img(image_path, target_size=self.image_size)
        arr = img_to_array(image).astype("float32") / 255.0
        return np.expand_dims(arr, axis=0)

    def predict_probabilities(self, image_path: str) -> np.ndarray:
        model = self._ensure_model_loaded()
        X = self._load_preprocess_image(image_path)
        probs = model.predict(X, verbose=0)[0]
        return probs

    def prob_feature_names(self) -> list[str]:
        names = []
        for idx in sorted(self.class_labels.keys()):
            label = self.class_labels[idx].lower().replace(" ", "_")
            names.append(f"mri_prob_{label}")
        return names

    def to_feature_row(self, image_path: str, class_name: str | None = None) -> dict:
        probs = self.predict_probabilities(image_path)
        feature_names = self.prob_feature_names()
        row = {
            "mri_image_path": image_path,
            "mri_source_class": class_name,
            "mri_pred_class_idx": int(np.argmax(probs)),
        }
        for i, col in enumerate(feature_names):
            row[col] = float(probs[i])
        return row


def build_mri_feature_table(
    image_root: str,
    model_path: str,
    class_dirs: Iterable[str],
    output_csv: str,
) -> pd.DataFrame:
    extractor = MriFeatureExtractor(model_path=model_path)
    rows: list[dict] = []

    image_root_path = Path(image_root)
    for class_dir in class_dirs:
        class_path = image_root_path / class_dir
        if not class_path.exists():
            continue

        images = sorted(
            p for p in class_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for image_path in images:
            rows.append(extractor.to_feature_row(str(image_path), class_name=class_dir))

    if not rows:
        raise RuntimeError(
            "No MRI images were processed. Check image root and class directory names."
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MRI probability feature table.")
    parser.add_argument("--image-root", required=True, help="Root directory containing MRI class folders")
    parser.add_argument("--model-path", required=True, help="Path to trained MRI .h5 model")
    parser.add_argument(
        "--class-dirs",
        nargs="+",
        default=["No Impairment", "Very Mild Impairment", "Mild Impairment", "Moderate Impairment"],
        help="Class subfolders to include",
    )
    parser.add_argument(
        "--output-csv",
        default="classification/mri_feature_table.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = build_mri_feature_table(
        image_root=args.image_root,
        model_path=args.model_path,
        class_dirs=args.class_dirs,
        output_csv=args.output_csv,
    )
    print(f"Saved MRI feature table: {args.output_csv}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
