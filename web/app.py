from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from src.infer import predict_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt"
DEFAULT_CLASS_MAP_PATH = PROJECT_ROOT / "experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISPLAY_ALLOWED_TYPES = "jpg, jpeg, png, bmp, webp"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ARCHITECTURE = "simple_cnn"
TOP_K = 3
API_DISCLAIMER = "Baseline educational screening result, not a definitive diagnosis."


class WebPredictionError(ValueError):
    """Raised for user-facing web prediction errors."""


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        PLANTGUARD_CHECKPOINT_PATH=os.environ.get("PLANTGUARD_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)),
        PLANTGUARD_CLASS_MAP_PATH=os.environ.get("PLANTGUARD_CLASS_MAP_PATH", str(DEFAULT_CLASS_MAP_PATH)),
    )
    if config:
        app.config.update(config)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_upload_too_large(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Upload is too large. The maximum file size is 5 MB."}), 413
        return render_index("Upload is too large. The maximum file size is 5 MB."), 413

    @app.get("/")
    def index():
        return render_index()

    @app.get("/healthz")
    def healthz():
        return jsonify({"service": "plantguard", "status": "ok"})

    @app.post("/predict")
    def predict():
        try:
            uploaded = request.files.get("image")
            predictions = run_web_prediction(
                uploaded,
                checkpoint_path=Path(app.config["PLANTGUARD_CHECKPOINT_PATH"]),
                class_map_path=Path(app.config["PLANTGUARD_CLASS_MAP_PATH"]),
            )
        except WebPredictionError as exc:
            return render_index(str(exc)), 400
        return render_index(predictions=predictions), 200

    @app.post("/api/predict")
    def api_predict():
        try:
            uploaded = request.files.get("image")
            predictions = run_web_prediction(
                uploaded,
                checkpoint_path=Path(app.config["PLANTGUARD_CHECKPOINT_PATH"]),
                class_map_path=Path(app.config["PLANTGUARD_CLASS_MAP_PATH"]),
            )
        except WebPredictionError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"disclaimer": API_DISCLAIMER, "predictions": predictions}), 200

    return app


def render_index(error: str | None = None, predictions: list[dict[str, float | int | str]] | None = None) -> str:
    return render_template(
        "index.html",
        error=error,
        predictions=predictions or [],
        allowed_types=DISPLAY_ALLOWED_TYPES,
        max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
    )


def run_web_prediction(uploaded_file, checkpoint_path: Path, class_map_path: Path):
    if uploaded_file is None or not uploaded_file.filename:
        raise WebPredictionError("Please choose an image file before requesting a prediction.")

    filename = secure_filename(uploaded_file.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise WebPredictionError(f"Unsupported file type. Please upload one of: {allowed}.")

    if not checkpoint_path.exists() or not class_map_path.exists():
        raise WebPredictionError(
            "Model artifact is missing. Recreate the baseline checkpoint with "
            "`python src\\train.py --config experiments\\configs\\plantvillage_baseline_simple_cnn.yaml`, "
            "then try again."
        )

    temp_path: Path | None = None
    try:
        _validate_uploaded_image(uploaded_file)
        with tempfile.NamedTemporaryFile(prefix="plantguard-upload-", suffix=extension, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            uploaded_file.save(temp_file)

        return predict_image(
            checkpoint_path=checkpoint_path,
            class_map_path=class_map_path,
            image_path=temp_path,
            architecture=ARCHITECTURE,
            top_k=TOP_K,
            device_name="cpu",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise WebPredictionError(f"Prediction failed: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _validate_uploaded_image(uploaded_file) -> None:
    try:
        uploaded_file.stream.seek(0)
        with Image.open(uploaded_file.stream) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise WebPredictionError("The uploaded file could not be read as an image.") from exc
    finally:
        uploaded_file.stream.seek(0)


app = create_app()
