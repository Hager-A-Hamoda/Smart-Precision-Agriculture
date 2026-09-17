"""
Smart Precision Agriculture - Leaf Disease Classifier (Streamlit)
==================================================================
Deploys ONLY the two disease-classification models from the project:

  - Apple Leaf Disease   -> best_model FF.keras
  - Tomato Leaf Disease  -> tomatoDisease\mobilenetv2_stage1_best.keras

This file is expected to live in disease_app/, one level below the repo
root where the two model files above actually sit.

Run with:
    streamlit run streamlit_app.py
"""

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
# app.py lives in disease_app/, one level below the repo root where the
# model files actually are.
REPO_ROOT = ROOT.parent

# Paths exactly as they sit inside the repo (relative to the repo root).
APPLE_DISEASE_PATH = "best_model FF.keras"
TOMATO_DISEASE_PATH = r"tomatoDisease\mobilenetv2_stage1_best.keras"

# Class order matches the original project's app.py exactly.
APPLE_DISEASE_CLASSES = [
    "Healthy",
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
]

TOMATO_DISEASE_CLASSES = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
]

st.set_page_config(
    page_title="Leaf Disease Classifier",
    page_icon="🌿",
    layout="centered",
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def find_model(relative_path: str) -> Path:
    """Resolve a model path relative to the repo root (one level above
    disease_app/). Falls back to a recursive search under the repo root by
    filename if it isn't found at the expected spot."""
    path = REPO_ROOT / relative_path
    if path.exists():
        return path
    filename = Path(relative_path).name
    matches = list(REPO_ROOT.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find '{filename}' at the expected path '{path}'. "
        f"Make sure disease_app/ sits directly inside the repo root, next "
        f"to 'best_model FF.keras' and the 'tomatoDisease/' folder."
    )


@st.cache_resource(show_spinner=False)
def load_apple_disease_model():
    path = find_model(APPLE_DISEASE_PATH)
    return keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner=False)
def load_tomato_disease_model():
    path = find_model(TOMATO_DISEASE_PATH)
    return keras.models.load_model(path, compile=False)


def pil_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def preprocess_for_shape(image: Image.Image, shape) -> np.ndarray:
    """
    Resize according to the Keras model's expected input shape and
    normalize to [0, 1]. Matches the original project's preprocessing.
    """
    h = int(shape[1] or 224)
    w = int(shape[2] or 224)
    image = pil_rgb(image).resize((w, h))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image, class_names):
    """
    Run a Keras image classifier and return (label, confidence, all_probs dict).
    Handles both a single-sigmoid-output (binary) model and a
    multi-class softmax model.
    """
    shape = model.inputs[0].shape
    x = preprocess_for_shape(image, shape)
    raw = np.asarray(model.predict(x, verbose=0))[0]

    if raw.size == 1:
        # Binary sigmoid output -> class 1 probability
        p1 = float(np.ravel(raw)[0])
        probs = np.array([1.0 - p1, p1])
        # Only makes sense if exactly 2 classes are provided
        names = class_names[:2] if len(class_names) >= 2 else ["Class 0", "Class 1"]
    else:
        probs = raw.astype(float)
        names = class_names

    idx = int(np.argmax(probs))
    label = names[idx] if idx < len(names) else f"Class {idx}"
    confidence = float(probs[idx])
    prob_dict = {names[i] if i < len(names) else f"Class {i}": float(probs[i])
                 for i in range(len(probs))}
    return label, confidence, prob_dict


def render_prediction_ui(model_loader, class_names, uploader_key, title):
    st.subheader(title)
    uploaded = st.file_uploader(
        "Upload a leaf image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key=uploader_key,
    )

    if uploaded is None:
        st.info("Upload an image to run a prediction.")
        return

    image = Image.open(io.BytesIO(uploaded.getvalue()))
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run Prediction", key=f"btn_{uploader_key}", type="primary"):
        with st.spinner("Loading model and predicting..."):
            try:
                model = model_loader()
                label, confidence, prob_dict = predict(model, image, class_names)
            except FileNotFoundError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        if label.lower() == "healthy":
            st.success(f"**Prediction: {label}**  ·  Confidence: {confidence:.2%}")
        else:
            st.warning(f"**Prediction: {label}**  ·  Confidence: {confidence:.2%}")

        st.markdown("**Class probabilities**")
        st.bar_chart(prob_dict)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🌿 Leaf Disease Classifier")
st.caption(
    "Apple & Tomato leaf disease detection — from the Smart Precision "
    "Agriculture project. Deploys the `best_model FF.keras` (apple) and "
    "`mobilenetv2_stage1_best.keras` (tomato) classifiers."
)

tab_apple, tab_tomato = st.tabs(["🍎 Apple Leaf Disease", "🍅 Tomato Leaf Disease"])

with tab_apple:
    render_prediction_ui(
        load_apple_disease_model,
        APPLE_DISEASE_CLASSES,
        uploader_key="apple_uploader",
        title="Apple Leaf Disease Detection",
    )

with tab_tomato:
    render_prediction_ui(
        load_tomato_disease_model,
        TOMATO_DISEASE_CLASSES,
        uploader_key="tomato_uploader",
        title="Tomato Leaf Disease Detection",
    )

with st.sidebar:
    st.header("About")
    st.write(
        "This app loads two Keras `.keras` models directly (no YOLO, no "
        "fruit classifier, no maturity models) and classifies a leaf "
        "image as **Healthy** or one of several disease classes."
    )
    st.markdown("**Apple classes:**")
    st.write(", ".join(APPLE_DISEASE_CLASSES))
    st.markdown("**Tomato classes:**")
    st.write(", ".join(TOMATO_DISEASE_CLASSES))
    st.divider()
    st.caption(
        "Model files are expected at (relative to the repo root, one "
        "level above this app):\n\n"
        "`best_model FF.keras`\n\n"
        "`tomatoDisease\\mobilenetv2_stage1_best.keras`"
    )