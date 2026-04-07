# CNN+CHA Fusion Artifacts (Required)

Place the following files in this folder for the endpoint:

- `POST /predict/cnn-cha-fusion`

Required files:

1. `cnn_cha_fusion_v2.keras`
2. `voice_scaler.pkl`
3. `voice_feature_columns.pkl`

Optional file:

1. `cnn_cha_fusion_v2_report.json` (used to load `best_threshold`; defaults to `0.5` if missing)

## Where to get these files

Copy from:

- `../cnn_cha_fusion/artifacts/`

into:

- `backend/model_artifacts/cnn_cha_fusion/`

## Note

If files are not found here, backend currently also checks fallback path:

- `../cnn_cha_fusion/artifacts/`

You can override artifact directory using env var:

- `CNN_CHA_ARTIFACT_DIR`
