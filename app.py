"""
app.py - Streamlit demo of the flood detector.

    streamlit run app.py
    streamlit run app.py -- --model-dir models --data-dir data    (overrides after "--")

Single image: upload one image; the input and its Grad-CAM evidence appear side by
side, with the verdict, the flood probability, the decision threshold and the
attention area underneath.
Batch: upload several images; a table of file, verdict and probability, downloadable
as CSV.
Sidebar: a threshold slider that starts at the tuned value from models/threshold.json.
Moving it shows the precision/recall trade-off live, measured on the validation set.

The model is loaded once per server (st.cache_resource), not on every interaction.
Unreadable files are reported with st.error instead of a traceback.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config  # noqa: E402  (src/ must be on the path first)
from dataset import decode, load_for_model, make_datasets  # noqa: E402
from evaluate import scores  # noqa: E402
from gradcam import attention_area, explain, overlay, resize_heatmap  # noqa: E402
from model import load_trained_model  # noqa: E402

UPLOAD_TYPES = ["jpg", "jpeg", "png"]


def parse_args():
    """Options after `--` on the streamlit command line; unknown ones are ignored."""
    parser = argparse.ArgumentParser(description="Streamlit flood detector")
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    return parser.parse_known_args()[0]


# ---------------------------------------------------------------------------
# Loading - cached, so it happens once per server, not on every click
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model ...")
def load_resources(model_dir):
    """The trained model and the tuned threshold record from models/threshold.json."""
    import json

    model = load_trained_model(os.path.join(model_dir, config.BEST_MODEL_FILE), compile=False)
    path = os.path.join(model_dir, config.THRESHOLD_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{config.rel(path)} not found - run python src/evaluate.py first.")
    with open(path) as handle:
        threshold = json.load(handle)
    model.predict(np.zeros((1, *config.IMG_SIZE, config.CHANNELS), np.float32), verbose=0)  # warm-up
    return model, threshold


@st.cache_resource(show_spinner="Scoring the validation set for the threshold slider ...")
def validation_predictions(_model, data_dir):
    """Validation labels and probabilities, computed once, so the slider can show precision
    and recall at any threshold. (The leading underscore tells Streamlit not to hash the model.)"""
    _, val_ds, _, _ = make_datasets(data_dir, validate=False)
    probs = _model.predict(val_ds, verbose=0).ravel()
    labels = np.concatenate([y.numpy().ravel() for _, y in val_ds]).astype(int)
    return labels, probs


# ---------------------------------------------------------------------------
# Inference - plain functions, testable without a browser
# ---------------------------------------------------------------------------
def classify(model, data):
    """Encoded image bytes -> verdict details. Raises ValueError for an unreadable file."""
    try:
        original = decode(data).numpy()                 # full image, for display
        crop, model_input = load_for_model(data)        # what the model actually sees
    except Exception as exc:                            # TensorFlow raises several types here
        raise ValueError(f"not a readable JPEG/PNG image ({type(exc).__name__})") from exc
    prob, cam = explain(model, model_input)             # probability + Grad-CAM, one pass
    heat = resize_heatmap(cam, crop.shape[:2])
    return {"prob": prob, "original": original, "crop": crop,
            "overlay": overlay(crop, heat), "attention": attention_area(heat, warn=False)}


def classify_batch(model, files, threshold):
    """[(name, bytes)] -> (DataFrame of file / verdict / probability, [unreadable names])."""
    rows, inputs, bad = [], [], []
    for name, data in files:
        try:
            inputs.append(load_for_model(data)[1])
            rows.append(name)
        except Exception:                                # unreadable: reported, not raised
            bad.append(name)
    probs = model.predict(np.stack(inputs), verbose=0).ravel() if inputs else np.array([])
    table = pd.DataFrame({
        "file": rows,
        "verdict": ["FLOOD" if p >= threshold else "NO FLOOD" for p in probs],
        "flood_probability": np.round(probs, 4),
    })
    return table, bad


def verdict_html(prob, threshold):
    flood = prob >= threshold
    colour = config.DANGER if flood else config.OK
    text, risk = ("FLOOD DETECTED", "HIGH RISK") if flood else ("NO FLOOD", "LOW RISK")
    return (f"<div style='text-align:center;font-size:2.6rem;font-weight:800;color:{colour};"
            f"letter-spacing:0.04em'>{text}</div>"
            f"<div style='text-align:center;font-size:1.1rem;font-weight:700;color:{colour}'>{risk}</div>")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    st.set_page_config(page_title="Flood Detection System", page_icon="🌊", layout="wide")
    st.title("🌊 Flood Detection System")
    st.write("MobileNetV2 (ImageNet weights, fine-tuned on AIDER aerial images) classifies an "
             "image as **flood** or **no flood**, and Grad-CAM shows which part of the image "
             "drove the decision.")

    try:
        model, tuned = load_resources(args.model_dir)
    except Exception as exc:
        st.error(f"Could not load the model: {exc}\n\nTrain and evaluate first:  "
                 "`python src/train.py` then `python src/evaluate.py`.")
        st.stop()

    # Sidebar: the decision threshold, with its validation precision and recall.
    st.sidebar.header("Decision threshold")
    threshold = st.sidebar.slider("Flood if probability ≥", 0.05, 0.95, float(tuned["threshold"]), 0.01)
    st.sidebar.caption(f"Tuned value **{tuned['threshold']:.2f}**, chosen on the validation set: "
                       f"the lowest threshold with precision ≥ {tuned['min_precision']}.")
    try:
        labels, probs = validation_predictions(model, args.data_dir)
        at = scores(labels, probs, threshold)
        st.sidebar.metric("Validation precision", f"{at['precision']:.3f}")
        st.sidebar.metric("Validation recall", f"{at['recall']:.3f}")
        st.sidebar.caption(f"At {threshold:.2f}: {at['fn']} of {at['tp'] + at['fn']} validation floods "
                           f"missed, {at['fp']} false alarms among {at['tn'] + at['fp']} non-floods.")
    except Exception as exc:
        st.sidebar.warning(f"Validation set unavailable ({exc}); live precision/recall disabled.")

    single, batch = st.tabs(["Single image", "Batch"])

    with single:
        upload = st.file_uploader("Upload an image", type=UPLOAD_TYPES, key="single")
        if upload is not None:
            try:
                result = classify(model, upload.getvalue())
            except ValueError as exc:
                st.error(f"Could not read **{upload.name}**: {exc}.")
            else:
                left, right = st.columns(2)
                left.subheader("Input image")
                left.image(result["original"], width="stretch")
                right.subheader("Flood evidence (Grad-CAM)")
                right.image(result["overlay"], width="stretch",
                            caption="Conv_1 Grad-CAM on the centred crop the model sees "
                                    "(red = strongest flood evidence)")
                st.markdown(verdict_html(result["prob"], threshold), unsafe_allow_html=True)
                st.progress(min(max(result["prob"], 0.0), 1.0),
                            text=f"Flood probability: {100 * result['prob']:.2f}%")
                st.caption(f"Decision threshold: {threshold:.2f}  |  "
                           f"Attention area: {result['attention']:.1f}% of image")
                if result["attention"] > config.ATTENTION_WARN_PCT:
                    st.warning("Attention covers more than half of the image: the model is reacting to "
                               "the whole scene rather than localising water, so treat this verdict "
                               "with caution.")

    with batch:
        uploads = st.file_uploader("Upload several images", type=UPLOAD_TYPES,
                                   accept_multiple_files=True, key="batch")
        if uploads:
            table, bad = classify_batch(model, [(u.name, u.getvalue()) for u in uploads], threshold)
            if bad:
                st.error("Could not read: " + ", ".join(bad))
            if len(table):
                n_flood = int((table["verdict"] == "FLOOD").sum())
                st.write(f"**{len(table)} images** | {n_flood} flood | {len(table) - n_flood} no flood "
                         f"| threshold {threshold:.2f}")
                st.dataframe(table, hide_index=True)
                st.download_button("Download CSV", table.to_csv(index=False).encode(),
                                   file_name="batch_predictions.csv", mime="text/csv")

    st.caption("Trained on AIDER aerial imagery. Ground-level and CCTV images are untested - "
               "see README, 'Known limitations'.")


if __name__ == "__main__":
    main()
