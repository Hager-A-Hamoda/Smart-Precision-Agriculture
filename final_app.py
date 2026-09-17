from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

# app.py is inside disease_app/
# Therefore, BASE_DIR.parent is the repository root.

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

# ------------------------------------------------------------
# Fruit classification
# ------------------------------------------------------------

FRUIT_CLASSIFIER_PATH = (
    REPO_DIR
    / "apple_tomato_efficientnetb0_final2.keras"
)

# ------------------------------------------------------------
# Apple models
# ------------------------------------------------------------

APPLE_DETECTION_PATH = (
    REPO_DIR
    / "Apple Maturity Models"
    / "Apple detection model"
    / "apple_yolo11n_120_best.pt"
)

APPLE_MATURITY_PATH = (
    REPO_DIR
    / "Apple Maturity Models"
    / "Apple maturity model"
    / "apple_maturity_model.keras"
)

APPLE_DISEASE_PATH = (
    REPO_DIR
    / "best_model FF.keras"
)

# ------------------------------------------------------------
# Tomato models
# ------------------------------------------------------------

TOMATO_MATURITY_PATH = (
    REPO_DIR
    / "Tomato Maturity"
    / "tomato_maturity_best (1).pt"
)

TOMATO_DISEASE_PATH = (
    REPO_DIR
    / "tomatoDisease"
    / "mobilenetv2_stage1_best.keras"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# IMPORTANT:
# Class order must match the order used during training.

FRUIT_CLASSES = [
    "Apple",
    "Tomato",
]

APPLE_MATURITY_CLASSES = [
    "Unripe",
    "Semi-ripe",
    "Ripe",
]

TOMATO_MATURITY_CLASSES = [
    "Green",
    "Turning",
    "Ripe",
    "Overripe",
]

APPLE_DISEASE_CLASSES = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]

TOMATO_DISEASE_CLASSES = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
]


# ============================================================
# INPUT SIZES
# ============================================================

# These are the sizes used by the previous app configuration.
# Change them only if the corresponding training notebooks
# use different input sizes.

FRUIT_CLASSIFIER_SIZE = (224, 224)
APPLE_MATURITY_SIZE = (224, 224)
APPLE_DISEASE_SIZE = (224, 224)
TOMATO_DISEASE_SIZE = (224, 224)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_raw(
    image: Image.Image,
    size: tuple[int, int],
) -> np.ndarray:
    """
    Resize image and return raw RGB pixels in [0, 255].

    Use this when the model has its own Rescaling layer
    or was trained directly on pixel values.
    """
    image = image.convert("RGB").resize(size)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def preprocess_01(
    image: Image.Image,
    size: tuple[int, int],
) -> np.ndarray:
    """
    Resize image and normalize pixels to [0, 1].

    Use this only if the training notebook explicitly
    divided images by 255 before training.
    """
    image = image.convert("RGB").resize(size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def preprocess_for(
    image: Image.Image,
    size: tuple[int, int],
    method: str = "raw",
) -> np.ndarray:
    if method == "01":
        return preprocess_01(image, size)

    return preprocess_raw(image, size)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_keras_model(path_str: str):
    return tf.keras.models.load_model(path_str)


@st.cache_resource
def load_yolo_model(path_str: str):
    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed. "
            "Run: pip install ultralytics"
        )

    return YOLO(path_str)


# ============================================================
# OUTPUT INTERPRETATION
# ============================================================

def scores_to_prediction(
    raw_output: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Convert a Keras output into class probabilities.

    Supports:
    - Binary sigmoid output: [p]
    - Binary output: [p0, p1]
    - Softmax probabilities
    - Logits
    """

    raw_output = np.asarray(raw_output).squeeze()

    # --------------------------------------------------------
    # Single sigmoid output
    # --------------------------------------------------------

    if raw_output.ndim == 0:
        probability = float(raw_output)

        if not 0.0 <= probability <= 1.0:
            probability = float(
                tf.sigmoid(probability).numpy()
            )

        if len(class_names) != 2:
            raise ValueError(
                "Single output requires exactly 2 classes."
            )

        scores = np.array(
            [1.0 - probability, probability],
            dtype=np.float32,
        )

    elif raw_output.shape == (1,) and len(class_names) == 2:
        probability = float(raw_output[0])

        if not 0.0 <= probability <= 1.0:
            probability = float(
                tf.sigmoid(probability).numpy()
            )

        scores = np.array(
            [1.0 - probability, probability],
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Multi-class output
    # --------------------------------------------------------

    else:
        scores = raw_output.astype(np.float32).flatten()

        if len(scores) != len(class_names):
            raise ValueError(
                f"Model output has {len(scores)} values, "
                f"but {len(class_names)} classes were configured."
            )

        # If values are not a valid probability distribution,
        # interpret them as logits.
        if (
            np.any(scores < 0)
            or not np.isclose(
                np.sum(scores),
                1.0,
                atol=1e-3,
            )
        ):
            scores = tf.nn.softmax(scores).numpy()

        scores = scores / np.sum(scores)

    best_index = int(np.argmax(scores))

    return {
        "label": class_names[best_index],
        "confidence": float(scores[best_index]),
        "scores": list(
            zip(class_names, scores.tolist())
        ),
    }


# ============================================================
# KERAS CLASSIFICATION
# ============================================================

def predict_keras_classifier(
    image: Image.Image,
    model_path: Path,
    class_names: list[str],
    input_size: tuple[int, int],
    preprocess_method: str = "raw",
) -> dict:

    if not model_path.exists():
        return {
            "ready": False,
            "error": f"Model file not found:\n{model_path}",
        }

    try:
        model = load_keras_model(str(model_path))

        batch = preprocess_for(
            image,
            input_size,
            preprocess_method,
        )

        raw_output = model.predict(
            batch,
            verbose=0,
        )[0]

        result = scores_to_prediction(
            raw_output,
            class_names,
        )

        result["ready"] = True
        result["model_path"] = str(model_path)

        return result

    except Exception as error:
        return {
            "ready": False,
            "error": str(error),
            "model_path": str(model_path),
        }


# ============================================================
# YOLO PREDICTION
# ============================================================

def predict_yolo(
    image: Image.Image,
    model_path: Path,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Supports both YOLO classification and detection models.

    Classification:
        result.probs

    Detection:
        result.boxes
    """

    if not model_path.exists():
        return {
            "ready": False,
            "error": f"Model file not found:\n{model_path}",
        }

    if YOLO is None:
        return {
            "ready": False,
            "error": (
                "ultralytics is not installed.\n"
                "Run: pip install ultralytics"
            ),
        }

    try:
        model = load_yolo_model(str(model_path))

        results = model.predict(
            source=np.asarray(image.convert("RGB")),
            verbose=False,
        )

        if not results:
            return {
                "ready": False,
                "error": "YOLO returned no results.",
            }

        result = results[0]

        # ----------------------------------------------------
        # YOLO CLASSIFICATION
        # ----------------------------------------------------

        if result.probs is not None:

            probabilities = (
                result.probs.data.cpu().numpy()
            )

            names = result.names

            if class_names is None:
                class_names = [
                    names[i]
                    for i in range(len(probabilities))
                ]

            if len(class_names) != len(probabilities):
                return {
                    "ready": False,
                    "error": (
                        f"YOLO returned "
                        f"{len(probabilities)} classes, "
                        f"but {len(class_names)} "
                        f"were configured."
                    ),
                }

            prediction = scores_to_prediction(
                probabilities,
                class_names,
            )

            prediction["ready"] = True
            prediction["model_path"] = str(model_path)
            prediction["model_type"] = "classification"

            return prediction

        # ----------------------------------------------------
        # YOLO DETECTION
        # ----------------------------------------------------

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            boxes = result.boxes

            best_index = int(
                np.argmax(
                    boxes.conf.cpu().numpy()
                )
            )

            confidence = float(
                boxes.conf[best_index].cpu().item()
            )

            class_id = int(
                boxes.cls[best_index].cpu().item()
            )

            label = result.names.get(
                class_id,
                str(class_id),
            )

            detected_objects = []

            for i in range(len(boxes)):

                detected_class_id = int(
                    boxes.cls[i].cpu().item()
                )

                detected_objects.append({
                    "label": result.names.get(
                        detected_class_id,
                        str(detected_class_id),
                    ),
                    "confidence": float(
                        boxes.conf[i].cpu().item()
                    ),
                })

            return {
                "ready": True,
                "model_path": str(model_path),
                "model_type": "detection",
                "label": label,
                "confidence": confidence,
                "scores": [(label, confidence)],
                "detected_objects": detected_objects,
            }

        return {
            "ready": False,
            "error": "No objects were detected by YOLO.",
            "model_path": str(model_path),
        }

    except Exception as error:
        return {
            "ready": False,
            "error": str(error),
            "model_path": str(model_path),
        }


# ============================================================
# UI HELPERS
# ============================================================

def render_scores(scores):

    if not scores:
        return

    st.markdown("### Class probabilities")

    sorted_scores = sorted(
        scores,
        key=lambda item: item[1],
        reverse=True,
    )

    for label, probability in sorted_scores:

        st.write(
            f"**{label}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))


def render_prediction(result: dict):

    if not result.get("ready", False):

        st.error(
            "Prediction failed or model is unavailable."
        )

        if result.get("error"):
            st.code(result["error"])

        return

    st.success(
        f"Prediction: {result['label']}"
    )

    st.metric(
        "Confidence",
        f"{result['confidence'] * 100:.2f}%",
    )

    render_scores(
        result.get("scores", [])
    )

    if result.get("model_type") == "detection":

        st.markdown("### Detected objects")

        for obj in result.get(
            "detected_objects",
            [],
        ):
            st.write(
                f"- {obj['label']}: "
                f"{obj['confidence'] * 100:.2f}%"
            )


def render_model_status():

    st.sidebar.markdown("## Model status")

    models = {
        "Fruit classification": FRUIT_CLASSIFIER_PATH,
        "Apple detection": APPLE_DETECTION_PATH,
        "Apple maturity": APPLE_MATURITY_PATH,
        "Apple disease": APPLE_DISEASE_PATH,
        "Tomato maturity": TOMATO_MATURITY_PATH,
        "Tomato disease": TOMATO_DISEASE_PATH,
    }

    for name, path in models.items():

        if path.exists():
            st.sidebar.success(
                f"{name}: Connected"
            )
        else:
            st.sidebar.error(
                f"{name}: Missing"
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🍎 FruitVision AI")

    st.write(
        "Smart Precision Agriculture"
    )

    render_model_status()

    st.divider()

    st.caption(
        "All model paths are resolved relative "
        "to the repository root."
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title("🍎 FruitVision AI")

st.subheader(
    "Smart Precision Agriculture"
)

st.write(
    "Classify fruits, estimate maturity, "
    "and screen leaves for disease."
)


# ============================================================
# FRUIT SELECTION
# ============================================================

fruit = st.selectbox(
    "Select fruit",
    ["Apple", "Tomato"],
)


# ============================================================
# TASK SELECTION
# ============================================================

task = st.radio(
    "Choose a task",
    [
        "Maturity",
        "Disease",
        "Detection",
    ],
    horizontal=True,
)


# ============================================================
# MATURITY
# ============================================================

if task == "Maturity":

    st.markdown("## Fruit maturity")

    fruit_image_file = st.file_uploader(
        "Upload a fruit image",
        type=["jpg", "jpeg", "png", "webp"],
        key="maturity_upload",
    )

    if fruit_image_file is not None:

        fruit_image = Image.open(
            fruit_image_file
        ).convert("RGB")

        st.image(
            fruit_image,
            caption="Uploaded fruit",
            use_container_width=True,
        )

        if fruit == "Apple":

            st.markdown("### Apple maturity")

            result = predict_keras_classifier(
                image=fruit_image,
                model_path=APPLE_MATURITY_PATH,
                class_names=APPLE_MATURITY_CLASSES,
                input_size=APPLE_MATURITY_SIZE,
                preprocess_method="raw",
            )

        else:

            st.markdown("### Tomato maturity")

            result = predict_yolo(
                image=fruit_image,
                model_path=TOMATO_MATURITY_PATH,
                class_names=TOMATO_MATURITY_CLASSES,
            )

        render_prediction(result)


# ============================================================
# DISEASE
# ============================================================

elif task == "Disease":

    st.markdown("## Leaf disease screening")

    st.info(
        "Upload a clear image of one leaf. "
        "Disease models should receive a leaf image, "
        "not a whole fruit image."
    )

    leaf_image_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        key="disease_upload",
    )

    if leaf_image_file is not None:

        leaf_image = Image.open(
            leaf_image_file
        ).convert("RGB")

        st.image(
            leaf_image,
            caption="Uploaded leaf",
            use_container_width=True,
        )

        if fruit == "Apple":

            st.markdown("### Apple leaf disease")

            result = predict_keras_classifier(
                image=leaf_image,
                model_path=APPLE_DISEASE_PATH,
                class_names=APPLE_DISEASE_CLASSES,
                input_size=APPLE_DISEASE_SIZE,
                preprocess_method="raw",
            )

        else:

            st.markdown("### Tomato leaf disease")

            result = predict_keras_classifier(
                image=leaf_image,
                model_path=TOMATO_DISEASE_PATH,
                class_names=TOMATO_DISEASE_CLASSES,
                input_size=TOMATO_DISEASE_SIZE,
                preprocess_method="raw",
            )

        render_prediction(result)


# ============================================================
# DETECTION
# ============================================================

elif task == "Detection":

    st.markdown("## Fruit detection")

    detection_image_file = st.file_uploader(
        "Upload an image for detection",
        type=["jpg", "jpeg", "png", "webp"],
        key="detection_upload",
    )

    if detection_image_file is not None:

        detection_image = Image.open(
            detection_image_file
        ).convert("RGB")

        st.image(
            detection_image,
            caption="Uploaded image",
            use_container_width=True,
        )

        if fruit == "Apple":

            st.markdown("### Apple detection")

            result = predict_yolo(
                image=detection_image,
                model_path=APPLE_DETECTION_PATH,
            )

            render_prediction(result)

        else:

            st.info(
                "No Tomato detection model was provided "
                "in the repository tree."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FruitVision AI — Smart Precision Agriculture"
)