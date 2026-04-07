"""
FastAPI backend for Control / MCI classification from .cha files.

Endpoints
---------
POST /predict          Upload a single .cha file  → prediction JSON
POST /predict/batch    Upload multiple .cha files  → list of predictions
GET  /health           Health-check / model info
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from predict import (
    predict_from_cha_lines,
    predict_multimodal_from_inputs,
    predict_cnn_cha_from_inputs,
    get_artifacts,
    get_multimodal_artifacts,
    get_cnn_cha_artifacts,
    DIAGNOSIS_NAMES,
    MULTIMODAL_LABELS,
)

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MCI / Control Classifier",
    description=(
        "Upload a CHAT (.cha) transcript file and get a prediction "
        "of whether the patient is **Control** or **MCI** based on "
        "speech-pause features extracted from the file."
    ),
    version="1.0.0",
)

# Allow all origins so any frontend can call this API.
# Tighten this for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Startup event — pre-load model so the first request isn't slow
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_model_on_startup():
    try:
        get_artifacts()
        print("Model artifacts loaded successfully.")
    except Exception as e:
        # Don't crash the server; the /health endpoint will report the issue.
        print(f"WARNING: {e}")

    try:
        get_cnn_cha_artifacts()
        print("CNN+CHA artifacts loaded successfully.")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"WARNING: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Return server status and model info."""
    payload = {}

    try:
        model, scaler, feature_cols = get_artifacts()
        payload["voice_model"] = {
            "status": "ok",
            "model_type": type(model).__name__,
            "features": feature_cols,
            "classes": DIAGNOSIS_NAMES,
        }
    except Exception as e:
        payload["voice_model"] = {"status": "error", "detail": str(e)}

    try:
        multimodal_model, multimodal_features = get_multimodal_artifacts()
        payload["multimodal_model"] = {
            "status": "ok",
            "model_type": type(multimodal_model).__name__,
            "features": multimodal_features,
            "classes": MULTIMODAL_LABELS,
        }
    except Exception as e:
        payload["multimodal_model"] = {"status": "not_ready", "detail": str(e)}

    try:
        cnn_model, _, cnn_features, cnn_threshold = get_cnn_cha_artifacts()
        payload["cnn_cha_fusion_model"] = {
            "status": "ok",
            "model_type": type(cnn_model).__name__,
            "features": cnn_features,
            "classes": {0: "Control", 1: "MCI"},
            "default_threshold": cnn_threshold,
        }
    except (FileNotFoundError, RuntimeError) as e:
        payload["cnn_cha_fusion_model"] = {"status": "not_ready", "detail": str(e)}

    payload["status"] = "ok"
    return payload


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    """
    Upload a single **.cha** file and receive a diagnosis prediction.

    Returns JSON with `predicted_diagnosis`, `confidence`, per-class
    `probabilities`, and the extracted `features`.
    """
    if not file.filename.endswith(".cha"):
        raise HTTPException(status_code=400, detail="Only .cha files are accepted.")

    content = await file.read()

    try:
        lines = content.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    result = predict_from_cha_lines(lines, filename=file.filename)

    if result is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract features from the uploaded file. "
                "Make sure it is a valid .cha transcript with *PAR: and %wor: lines."
            ),
        )

    return result


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Upload **multiple .cha files** at once and get predictions for each.

    Returns a JSON list of prediction objects (same schema as `/predict`).
    Files that fail feature extraction are included with `"error"` instead of
    prediction fields.
    """
    results = []

    for file in files:
        if not file.filename.endswith(".cha"):
            results.append({"filename": file.filename, "error": "Not a .cha file — skipped."})
            continue

        content = await file.read()

        try:
            lines = content.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            results.append({"filename": file.filename, "error": "Encoding error."})
            continue

        result = predict_from_cha_lines(lines, filename=file.filename)

        if result is None:
            results.append({"filename": file.filename, "error": "Feature extraction failed."})
        else:
            results.append(result)

    return results


@app.post("/predict/multimodal")
async def predict_multimodal(
    cha_file: UploadFile = File(...),
    mri_file: UploadFile = File(...),
):
    """
    Upload one .cha transcript and one MRI image for fused multimodal prediction.
    """
    if not cha_file.filename.endswith(".cha"):
        raise HTTPException(status_code=400, detail="cha_file must be a .cha file.")

    valid_img_ext = (".jpg", ".jpeg", ".png")
    if not mri_file.filename.lower().endswith(valid_img_ext):
        raise HTTPException(
            status_code=400,
            detail="mri_file must be one of: .jpg, .jpeg, .png",
        )

    cha_content = await cha_file.read()
    mri_content = await mri_file.read()

    try:
        cha_lines = cha_content.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="cha_file must be UTF-8 encoded text.")

    try:
        result = predict_multimodal_from_inputs(
            cha_lines=cha_lines,
            mri_image_bytes=mri_content,
            cha_filename=cha_file.filename,
            mri_filename=mri_file.filename,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract transcript voice features from cha_file. "
                "Ensure the file has valid *PAR: and %wor: lines."
            ),
        )

    return result


@app.post("/predict/cnn-cha-fusion")
async def predict_cnn_cha_fusion(
    cha_file: UploadFile = File(...),
    mri_file: UploadFile = File(...),
    threshold_override: float | None = Form(default=None),
):
    """
    Upload one .cha and one MRI image for CNN+CHA binary prediction.

    Returns:
    {
      "P(Control)": 0.0247,
      "P(MCI)": 0.9753,
      "Prediction": "MCI"
    }
    """
    if not cha_file.filename.endswith(".cha"):
        raise HTTPException(status_code=400, detail="cha_file must be a .cha file.")

    valid_img_ext = (".jpg", ".jpeg", ".png")
    if not mri_file.filename.lower().endswith(valid_img_ext):
        raise HTTPException(
            status_code=400,
            detail="mri_file must be one of: .jpg, .jpeg, .png",
        )

    cha_content = await cha_file.read()
    mri_content = await mri_file.read()

    try:
        cha_lines = cha_content.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="cha_file must be UTF-8 encoded text.")

    try:
        result = predict_cnn_cha_from_inputs(
            cha_lines=cha_lines,
            mri_image_bytes=mri_content,
            cha_filename=cha_file.filename,
            mri_filename=mri_file.filename,
            threshold_override=threshold_override,
        )
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract transcript voice features from cha_file. "
                "Ensure the file has valid *PAR: and %wor: lines."
            ),
        )

    return result
