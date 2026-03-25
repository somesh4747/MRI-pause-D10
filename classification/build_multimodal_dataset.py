"""Create a multimodal training table by fusing voice rows with MRI probability features."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
from dataclasses import dataclass

import pandas as pd


VOICE_TO_MRI_CLASS = {
    0: "No Impairment",
    1: "Very Mild Impairment",
}

MRI_CLASS_TO_TARGET = {
    "Mild Impairment": 0,
    "Moderate Impairment": 1,
    "No Impairment": 2,
    "Very Mild Impairment": 3,
}


@dataclass
class SplitConfig:
    test_size: float = 0.2
    seed: int = 42


def canonicalize_patient_id(patient_id: str) -> str:
    s = str(patient_id)
    s = re.sub(r"^(control_aug|mci_aug)_\d+_", "", s)
    s = re.sub(r"-v\d+$", "", s)
    return s


def assign_group_split(group_id: str, test_size: float, seed: int) -> str:
    # Stable per-group split assignment so augmented variants stay together.
    key = f"{group_id}|{seed}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest(), 16) % 10_000
    threshold = int(test_size * 10_000)
    return "test" if bucket < threshold else "train"


def sample_mri_row(group: pd.DataFrame, rng: random.Random) -> pd.Series:
    idx = rng.randrange(len(group))
    return group.iloc[idx]


def build_multimodal_dataset(
    voice_csv: str,
    mri_feature_csv: str,
    output_csv: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    voice_df = pd.read_csv(voice_csv)
    mri_df = pd.read_csv(mri_feature_csv)

    required_voice_cols = {
        "patient_id",
        "diagnosis",
        "diagnosis_name",
        "pause_count",
        "total_speech_time",
        "total_pause_time",
        "mean_word_duration",
        "speech_rate_wpm",
        "pause_per_word_ratio",
    }
    missing_voice = required_voice_cols - set(voice_df.columns)
    if missing_voice:
        raise ValueError(f"voice csv missing columns: {sorted(missing_voice)}")

    if "mri_source_class" not in mri_df.columns:
        raise ValueError("mri feature csv missing required column: mri_source_class")

    rng = random.Random(seed)

    voice_df = voice_df.copy()
    voice_df["canonical_patient_id"] = voice_df["patient_id"].apply(canonicalize_patient_id)
    voice_df["target_mri_class"] = voice_df["diagnosis"].map(VOICE_TO_MRI_CLASS)

    if voice_df["target_mri_class"].isna().any():
        unknown = sorted(voice_df[voice_df["target_mri_class"].isna()]["diagnosis"].unique())
        raise ValueError(f"unsupported voice diagnosis values for mapping: {unknown}")

    mri_grouped = {k: v.reset_index(drop=True) for k, v in mri_df.groupby("mri_source_class")}

    rows = []
    mri_prob_cols = [c for c in mri_df.columns if c.startswith("mri_prob_")]
    for _, vrow in voice_df.iterrows():
        target_class = vrow["target_mri_class"]
        if target_class not in mri_grouped:
            raise ValueError(
                f"no MRI rows found for class '{target_class}'. "
                "Check class folder names and MRI feature table."
            )

        mri_row = sample_mri_row(mri_grouped[target_class], rng)

        fused = vrow.to_dict()
        for c in mri_prob_cols:
            fused[c] = float(mri_row[c])
        fused["mri_image_path"] = mri_row.get("mri_image_path")
        fused["mri_source_class"] = mri_row.get("mri_source_class")
        fused["pairing_strategy"] = "diagnosis_level_random"
        fused["target_4class"] = MRI_CLASS_TO_TARGET[target_class]
        fused["split"] = assign_group_split(
            group_id=fused["canonical_patient_id"],
            test_size=test_size,
            seed=seed,
        )
        rows.append(fused)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal fusion dataset CSV.")
    parser.add_argument(
        "--voice-csv",
        default="_training_dataset.csv",
        help="Voice feature CSV path",
    )
    parser.add_argument(
        "--mri-feature-csv",
        default="classification/mri_feature_table.csv",
        help="MRI feature CSV path",
    )
    parser.add_argument(
        "--output-csv",
        default="classification/multimodal_training_dataset.csv",
        help="Output multimodal CSV path",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_multimodal_dataset(
        voice_csv=args.voice_csv,
        mri_feature_csv=args.mri_feature_csv,
        output_csv=args.output_csv,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"Saved multimodal dataset: {args.output_csv}")
    print(df["target_4class"].value_counts(dropna=False).sort_index())
    print(df["split"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
