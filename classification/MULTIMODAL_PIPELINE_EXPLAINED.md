# Multimodal Pipeline Explained (MRI + Voice)

## What this pipeline does
This workflow builds a multimodal classifier by combining:
1. Voice features from `_training_dataset.csv`
2. MRI probability features extracted from MRI images using a pre-trained CNN `.h5` model

The final model is an early-fusion Random Forest trained on the combined feature table.

---

## Main files involved
- [classification/multimodal_full_pipeline.ipynb](classification/multimodal_full_pipeline.ipynb): orchestrates the full workflow step by step
- [classification/mri_feature_extractor.py](classification/mri_feature_extractor.py): loads MRI model, preprocesses images, outputs MRI probability features
- [classification/build_multimodal_dataset.py](classification/build_multimodal_dataset.py): fuses voice rows with MRI feature rows
- [classification/train_multimodal_random_forest.py](classification/train_multimodal_random_forest.py): trains and evaluates the multimodal Random Forest

---

## How MRI images are converted into features

### 1) Where MRI images come from
In the notebook, MRI images are read from:
- `Alzheimers-Disease-Classification/Combined Dataset/train`

Expected class folders:
- `No Impairment`
- `Very Mild Impairment`
- `Mild Impairment`
- `Moderate Impairment`

Only image files with extensions `.jpg`, `.jpeg`, `.png` are processed.

### 2) MRI model loading
`MriFeatureExtractor` in [classification/mri_feature_extractor.py](classification/mri_feature_extractor.py) loads the `.h5` model from the configured path.

It includes compatibility logic:
- First tries normal Keras load
- If loading fails due to old/new config key mismatch (example: `batch_shape`, `optional`), it sanitizes the model config and reloads weights
- If needed, it tries fallback model filenames in the same folder

This is done so MRI feature extraction works even when model files were saved in different TensorFlow/Keras versions.

### 3) Image preprocessing before prediction
For each image:
1. Load and resize to `224 x 224`
2. Convert to numeric array
3. Normalize pixels by dividing by `255.0`
4. Add batch dimension to shape `(1, 224, 224, channels)`

### 4) Getting MRI probability features
The MRI CNN predicts a probability vector for each image.
Those probabilities are saved as numeric features:
- `mri_prob_mild_impairment`
- `mri_prob_moderate_impairment`
- `mri_prob_no_impairment`
- `mri_prob_very_mild_impairment`

Also saved:
- `mri_pred_class_idx` (argmax predicted class index)
- `mri_image_path`
- `mri_source_class`

All image rows are written to:
- `classification/mri_feature_table.csv`

So, MRI is not represented by raw pixels at fusion time. It is represented by compact model-output probabilities.

---

## How multimodal training data is built

### 1) Voice schema check
Notebook validates that voice CSV includes required columns such as:
- `patient_id`
- `diagnosis`
- pause/speech timing features

### 2) Label mapping used for fusion
In [classification/build_multimodal_dataset.py](classification/build_multimodal_dataset.py):
- Voice diagnosis `0` (Control) maps to MRI class `No Impairment`
- Voice diagnosis `1` (MCI) maps to MRI class `Very Mild Impairment`

### 3) Row-by-row fusion process
For each voice row:
1. Determine mapped MRI class
2. Randomly sample one MRI feature row from that class
3. Copy MRI probability columns into the voice row
4. Add metadata (`canonical_patient_id`, `mri_source_class`, `pairing_strategy`, etc.)
5. Assign deterministic group-aware split (`train` or `test`) using hash of patient group + seed

Output:
- `classification/multimodal_training_dataset.csv`

Important: This is diagnosis-level random pairing, not guaranteed one-to-one patient-level MRI+voice pairing.

---

## How model training works

Training code is in [classification/train_multimodal_random_forest.py](classification/train_multimodal_random_forest.py).

### 1) Feature selection
Model uses all columns except metadata columns and the target column.
Target column:
- `target_4class`

### 2) Split handling
- Train on rows where `split == train`
- Test on rows where `split == test`
- Group key: `canonical_patient_id`

### 3) Model pipeline
Pipeline:
1. `StandardScaler`
2. `RandomForestClassifier` with class balancing and fixed seed

### 4) Validation strategy
Cross-validation uses `StratifiedGroupKFold` so the same patient group does not leak across folds.

Metrics collected:
- Accuracy
- Macro F1
- Weighted F1
- Balanced Accuracy

### 5) Final fit + test evaluation
After CV, model is fit on full training split and evaluated on test split with:
- Classification report
- Confusion matrix

---

## Saved artifacts
After training, these are saved under `model_artifacts/multimodal`:
- `multimodal_rf_model.pkl`
- `multimodal_feature_names.pkl`
- `multimodal_training_report.json`

Notebook then copies them to backend path:
- `backend/model_artifacts/multimodal`

This allows backend inference code to load the same trained model and exact feature order.

---

## End-to-end summary
1. Load voice table
2. Run MRI CNN on every MRI image to generate probability features
3. Build fused dataset (voice + MRI probabilities)
4. Do group-safe split
5. Train early-fusion Random Forest with group-aware CV
6. Evaluate on held-out test split
7. Save and export artifacts for backend inference

That is how MRI images are "taken" and turned into model-ready features in this project.