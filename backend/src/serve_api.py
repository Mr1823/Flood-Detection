"""
serve_api.py - a small local API so the React dashboard can run live predictions.

The dashboard is otherwise static: it reads numbers the pipeline already produced.
Prediction is the one thing a browser cannot do, because the Keras model has to run
somewhere. This serves exactly that, on localhost, reusing the same functions the
evaluation scripts use - so a prediction here and a prediction in the pipeline are the
same computation, not two implementations that can drift.

    python src/serve_api.py                 # http://127.0.0.1:8000
    python src/serve_api.py --port 8100

Endpoints
    GET  /api/health      model, threshold, device
    POST /api/predict     multipart "file" -> probability, label, Grad-CAM overlay
"""
import argparse
import base64
import io
import os

from contextlib import asynccontextmanager

import numpy as np

import config  # prints the TensorFlow version and devices; imported before TensorFlow
from dataset import load_for_model
from gradcam import attention_area, explain, overlay, resize_heatmap
from model import load_trained_model

MAX_UPLOAD_BYTES = 12 * 1024 * 1024          # a 224x224 crop is all we keep; cap the upload
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

STATE = {}


def load_threshold(model_dir):
    """The deployed decision threshold, from the file evaluate.py writes."""
    import json

    path = os.path.join(model_dir, config.THRESHOLD_FILE)
    if not os.path.isfile(path):
        raise SystemExit(f"[serve] MISSING: {path}\n"
                         f"[serve] Produce it first with: python src/evaluate.py")
    with open(path) as handle:
        return json.load(handle)


def png_data_uri(array):
    """uint8 HxWx3 -> a data: URI the browser can put straight in an <img src>."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@asynccontextmanager
async def lifespan(app):
    """Load the model once, at startup, not per request."""
    model_dir = app.state.model_dir
    STATE["threshold_record"] = load_threshold(model_dir)
    STATE["threshold"] = STATE["threshold_record"]["threshold"]
    model_path = os.path.join(model_dir, config.BEST_MODEL_FILE)
    STATE["model_path"] = config.rel(model_path)
    STATE["model"] = load_trained_model(model_path, compile=False)
    import tensorflow as tf

    STATE["device"] = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print(f"[serve] Model ready: {STATE['model_path']} at threshold {STATE['threshold']:.2f}")
    yield
    STATE.clear()


def build_app(model_dir):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Flood detection API", version="1.0.0", lifespan=lifespan)
    app.state.model_dir = model_dir

    # The Vite dev server proxies /api, but allow its origin directly too so the
    # dashboard works whether it is proxied or opened from the built files.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                       "http://localhost:4173", "http://127.0.0.1:4173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "model": STATE["model_path"],
            "threshold": STATE["threshold"],
            "threshold_rationale": STATE["threshold_record"]["rationale"],
            "device": STATE["device"],
            "class_names": list(config.CLASS_NAMES),
            "input_size": list(config.IMG_SIZE),
        }

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)):
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=415,
                                detail=f"Unsupported type {file.content_type!r}. "
                                       f"Send a JPEG, PNG or WebP image.")
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload.")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"Image is {len(raw) / 1e6:.1f} MB; the limit is "
                                       f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB.")
        try:
            display, model_input = load_for_model(raw)
        except Exception as error:                      # not an image, or a corrupt one
            raise HTTPException(status_code=400, detail=f"Could not read that image: {error}")

        # One forward pass gives both the probability and the Grad-CAM map.
        probability, cam = explain(STATE["model"], model_input)
        heatmap = resize_heatmap(cam, config.IMG_SIZE)
        threshold = STATE["threshold"]

        return {
            "p_flood": float(probability),
            "threshold": threshold,
            "label": config.CLASS_NAMES[int(probability >= threshold)],
            "is_flood": bool(probability >= threshold),
            "attention_area_pct": float(attention_area(heatmap, warn=False)),
            "attention_warn_pct": float(config.ATTENTION_WARN_PCT),
            "filename": file.filename,
            "image": png_data_uri(display),
            "overlay": png_data_uri(overlay(display, heatmap)),
            "note": ("Trained on AIDER aerial imagery. Ground-level and CCTV images are "
                     "untested - see the README, 'Known limitations'."),
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Local prediction API for the React dashboard.")
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    import uvicorn

    args = parse_args()
    uvicorn.run(build_app(args.model_dir), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
