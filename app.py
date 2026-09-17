import os
from pathlib import Path
from functools import lru_cache

import gradio as gr
import spaces
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO


# ============================================================
# Smart Precision Agriculture - Gradio / Hugging Face Space
# ============================================================

ROOT = Path(__file__).resolve().parent

# Model file names in the GitHub repository.
CLASSIFIER_NAME = "apple_tomato_efficientnetb0_final2.keras"

APPLE_DETECTOR_NAME = "apple_yolo11n_120_best.pt"
APPLE_MATURITY_NAME = "apple_maturity_model.keras"

TOMATO_MATURITY_NAME = "tomato_maturity_best (1).pt"

APPLE_DISEASE_NAME = "best_model FF.keras"
TOMATO_DISEASE_NAME = "mobilenetv2_stage1_best.keras"


# Class names used by the project.
APPLE_MATURITY_CLASSES = ["Unripe", "Semi-ripe", "Ripe"]

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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def find_model(filename: str) -> Path:
    """Find a model anywhere inside the Space repository."""
    matches = list(ROOT.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Model file was not found in the Space repository: {filename}"
        )

    return matches[0]


def pil_rgb(image):
    """Convert a Gradio/PIL image to RGB PIL."""
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    return Image.fromarray(image).convert("RGB")


def preprocess_for_shape(image, shape):
    """
    Resize according to a Keras input shape and normalize to [0, 1].

    The project models are image classifiers and use RGB images.
    """
    image = pil_rgb(image)

    h = int(shape[1] or 224)
    w = int(shape[2] or 224)

    arr = np.asarray(
        image.resize((w, h)),
        dtype=np.float32,
    ) / 255.0

    return np.expand_dims(arr, axis=0)


# ------------------------------------------------------------
# Lazy model loading
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_classifier():
    return keras.models.load_model(
        find_model(CLASSIFIER_NAME),
        compile=False,
    )


@lru_cache(maxsize=1)
def load_apple_detector():
    return YOLO(str(find_model(APPLE_DETECTOR_NAME)))


@lru_cache(maxsize=1)
def load_apple_maturity():
    return keras.models.load_model(
        find_model(APPLE_MATURITY_NAME),
        compile=False,
    )


@lru_cache(maxsize=1)
def load_tomato_maturity():
    return YOLO(str(find_model(TOMATO_MATURITY_NAME)))


@lru_cache(maxsize=1)
def load_apple_disease():
    return keras.models.load_model(
        find_model(APPLE_DISEASE_NAME),
        compile=False,
    )


@lru_cache(maxsize=1)
def load_tomato_disease():
    return keras.models.load_model(
        find_model(TOMATO_DISEASE_NAME),
        compile=False,
    )


# ------------------------------------------------------------
# 1. Apple / Tomato classification
# ------------------------------------------------------------

@spaces.GPU(duration=60)
def classifier_predict(image):
    """
    Run the Apple/Tomato classifier.

    For the project's binary sigmoid classifier:
        output = tomato probability
        apple probability = 1 - output
    """
    model = load_classifier()

    x = preprocess_for_shape(
        image,
        model.inputs[0].shape,
    )

    y = np.asarray(
        model.predict(x, verbose=0)
    )

    # Binary sigmoid model.
    if y.shape[-1] == 1:
        tomato_probability = float(y.reshape(-1)[0])
        apple_probability = 1.0 - tomato_probability

        if tomato_probability >= 0.5:
            return "Tomato", tomato_probability

        return "Apple", apple_probability

    # Fallback for a 2-class softmax model.
    idx = int(np.argmax(y[0]))

    names = getattr(
        model,
        "class_names",
        None,
    ) or ["Apple", "Tomato"]

    label = (
        str(names[idx])
        if idx < len(names)
        else str(idx)
    )

    return label, float(y[0][idx])


def classify_fruit(image):
    """Classify the uploaded fruit image."""
    if image is None:
        return (
            "",
            "Please upload a fruit image first.",
        )

    label, confidence = classifier_predict(image)

    result = (
        f"### Fruit Classification\n"
        f"**{label}**\n\n"
        f"Confidence: **{confidence:.2%}**"
    )

    return label, result


# ------------------------------------------------------------
# 2. Apple YOLO detection
# ------------------------------------------------------------

def apple_detection_and_crops(image):
    """
    Detect apples with YOLO and return:
      - bounding boxes
      - cropped apple images
    """
    detector = load_apple_detector()

    result = detector(
        pil_rgb(image),
        conf=0.25,
        verbose=False,
    )[0]

    boxes = []
    crops = []

    if result.boxes is None or len(result.boxes) == 0:
        return boxes, crops

    original = pil_rgb(image)

    for box in result.boxes:
        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .tolist()
        )

        conf = float(
            box.conf[0]
            .cpu()
            .item()
        )

        x1, y1, x2, y2 = [
            int(round(v))
            for v in xyxy
        ]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original.width, x2)
        y2 = min(original.height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append(
            (x1, y1, x2, y2, conf)
        )

        crops.append(
            original.crop(
                (x1, y1, x2, y2)
            )
        )

    return boxes, crops


def annotate_boxes(image, boxes):
    """Draw YOLO apple detections on the original image."""
    out = pil_rgb(image).copy()
    draw = ImageDraw.Draw(out)

    for i, (
        x1,
        y1,
        x2,
        y2,
        conf,
    ) in enumerate(boxes, start=1):

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="lime",
            width=4,
        )

        draw.text(
            (x1 + 5, y1 + 5),
            f"Apple {i}  {conf:.2f}",
            fill="lime",
        )

    return out


# ------------------------------------------------------------
# 3. Apple maturity
#    YOLO crop + original image -> EfficientNet
# ------------------------------------------------------------

def apple_maturity_predict(
    original_image,
    crop_image,
):
    """
    Run the two-input Apple maturity model.

    Expected project design:
        input 1 = YOLO apple crop
        input 2 = original classification image

    If the saved model input tensor names contain 'crop' and
    'original'/'full', those names are used automatically.
    Otherwise the project convention above is used.
    """
    model = load_apple_maturity()

    if len(model.inputs) != 2:
        raise ValueError(
            "Apple maturity model must have exactly 2 inputs, "
            f"but found {len(model.inputs)}."
        )

    shapes = [
        tuple(t.shape)
        for t in model.inputs
    ]

    names = [
        str(t.name).lower()
        for t in model.inputs
    ]

    crop_idx = next(
        (
            i
            for i, name in enumerate(names)
            if "crop" in name
        ),
        None,
    )

    original_idx = next(
        (
            i
            for i, name in enumerate(names)
            if (
                "original" in name
                or "full" in name
            )
        ),
        None,
    )

    inputs = [None, None]

    if (
        crop_idx is not None
        and original_idx is not None
        and crop_idx != original_idx
    ):
        inputs[crop_idx] = preprocess_for_shape(
            crop_image,
            shapes[crop_idx],
        )

        inputs[original_idx] = preprocess_for_shape(
            original_image,
            shapes[original_idx],
        )

    else:
        # Project convention:
        # first input = YOLO crop
        # second input = original image
        inputs[0] = preprocess_for_shape(
            crop_image,
            shapes[0],
        )

        inputs[1] = preprocess_for_shape(
            original_image,
            shapes[1],
        )

    prediction = np.asarray(
        model.predict(
            inputs,
            verbose=0,
        )
    )

    if (
        prediction.ndim != 2
        or prediction.shape[-1] != 3
    ):
        raise ValueError(
            "Unexpected Apple maturity output shape: "
            f"{prediction.shape}. "
            "The app expects 3 classes: "
            "Unripe, Semi-ripe, Ripe."
        )

    idx = int(
        np.argmax(prediction[0])
    )

    label = (
        APPLE_MATURITY_CLASSES[idx]
        if idx < len(
            APPLE_MATURITY_CLASSES
        )
        else f"Class {idx}"
    )

    confidence = float(
        prediction[0][idx]
    )

    return label, confidence


# ------------------------------------------------------------
# 4. Tomato maturity
# ------------------------------------------------------------

def tomato_maturity_predict(image):
    """
    Run the Tomato maturity .pt model.

    The .pt model is loaded through Ultralytics.
    This function expects a YOLO classification model and
    reads result.probs.
    """
    model = load_tomato_maturity()

    result = model(
        pil_rgb(image),
        verbose=False,
    )[0]

    if result.probs is None:
        raise ValueError(
            "The Tomato maturity .pt model did not return "
            "classification probabilities (result.probs)."
        )

    idx = int(
        result.probs.top1
    )

    confidence = float(
        result.probs.top1conf
    )

    names = result.names

    if isinstance(names, dict):
        label = names.get(
            idx,
            str(idx),
        )
    else:
        label = str(idx)

    return str(label), confidence


# ------------------------------------------------------------
# 5. Maturity workflow
# ------------------------------------------------------------

@spaces.GPU(duration=120)
def run_maturity(
    fruit_image,
    fruit,
):
    """Run the maturity model selected by fruit classification."""
    if fruit_image is None:
        return (
            None,
            "Please upload the fruit image first.",
        )

    if not fruit:
        return (
            None,
            "Please click 'Classify Fruit' first.",
        )

    # -------------------------
    # Apple
    # -------------------------
    if fruit.lower() == "apple":

        boxes, crops = (
            apple_detection_and_crops(
                fruit_image
            )
        )

        if not crops:
            return (
                fruit_image,
                "### Apple maturity\n"
                "No apple was detected by the "
                "Apple YOLO detector.",
            )

        annotated = annotate_boxes(
            fruit_image,
            boxes,
        )

        results = []

        for crop in crops:
            label, confidence = (
                apple_maturity_predict(
                    fruit_image,
                    crop,
                )
            )

            results.append(
                (label, confidence)
            )

        summary = "### Apple Maturity\n\n"

        for i, (
            label,
            confidence,
        ) in enumerate(
            results,
            start=1,
        ):
            summary += (
                f"**Apple {i}:** "
                f"{label} "
                f"({confidence:.2%})\n\n"
            )

        return annotated, summary

    # -------------------------
    # Tomato
    # -------------------------
    label, confidence = (
        tomato_maturity_predict(
            fruit_image
        )
    )

    return (
        fruit_image,
        "### Tomato Maturity\n\n"
        f"**{label}** — "
        f"{confidence:.2%}",
    )


# ------------------------------------------------------------
# 6. Leaf disease
# ------------------------------------------------------------

def keras_classifier_predict(
    model,
    image,
    class_names,
):
    """Run a Keras image classification model."""
    shape = model.inputs[0].shape

    x = preprocess_for_shape(
        image,
        shape,
    )

    prediction = np.asarray(
        model.predict(
            x,
            verbose=0,
        )
    )[0]

    # Binary sigmoid output.
    if prediction.size == 1:
        probability = float(
            np.ravel(prediction)[0]
        )

        idx = (
            1
            if probability >= 0.5
            else 0
        )

        label = class_names[idx]

        confidence = (
            probability
            if idx == 1
            else 1.0 - probability
        )

        return label, confidence

    # Multi-class softmax output.
    idx = int(
        np.argmax(prediction)
    )

    label = (
        class_names[idx]
        if idx < len(class_names)
        else f"Class {idx}"
    )

    confidence = float(
        prediction[idx]
    )

    return label, confidence


@spaces.GPU(duration=120)
def run_disease(
    leaf_image,
    fruit,
):
    """
    Select the disease model according to the first
    fruit classification result.
    """
    if leaf_image is None:
        return "Please upload a leaf image first."

    if not fruit:
        return (
            "Please click 'Classify Fruit' first "
            "so the correct leaf-disease model can be selected."
        )

    if fruit.lower() == "apple":
        label, confidence = (
            keras_classifier_predict(
                load_apple_disease(),
                leaf_image,
                APPLE_DISEASE_CLASSES,
            )
        )
    else:
        label, confidence = (
            keras_classifier_predict(
                load_tomato_disease(),
                leaf_image,
                TOMATO_DISEASE_CLASSES,
            )
        )

    return (
        f"### {fruit} Leaf Disease\n\n"
        f"**{label}** — "
        f"{confidence:.2%}"
    )


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------

with gr.Blocks(
    title="Smart Precision Agriculture",
) as demo:

    fruit_state = gr.State("")

    gr.Markdown(
        """
# 🌱 Smart Precision Agriculture

### Apple & Tomato
Upload a fruit image → classify it → choose:

- 🍎🍅 **Maturity**
- 🌿 **Leaf Disease**

For Apple maturity, the system uses:
**Apple YOLO detection → apple crop + original image → two-input EfficientNet**.
"""
    )

    # =========================
    # STEP 1
    # =========================

    gr.Markdown("## 1️⃣ Fruit Classification")

    with gr.Row():

        with gr.Column():

            fruit_image = gr.Image(
                type="pil",
                label="Upload Apple or Tomato",
            )

            classify_btn = gr.Button(
                "Classify Fruit",
                variant="primary",
            )

        with gr.Column():

            fruit_result = gr.Markdown(
                "Upload a fruit image and click "
                "**Classify Fruit**."
            )

    classify_btn.click(
        fn=classify_fruit,
        inputs=fruit_image,
        outputs=[
            fruit_state,
            fruit_result,
        ],
    )

    # =========================
    # STEP 2
    # =========================

    gr.Markdown("## 2️⃣ Choose the Task")

    with gr.Tab(
        "🍎🍅 Maturity"
    ):

        gr.Markdown(
            """
The maturity model is selected automatically from
the fruit classification.

**Apple:** YOLO detects/crops apples, then the two-input
EfficientNet receives the crop and the original image.

**Tomato:** the Tomato maturity `.pt` model is used.
"""
        )

        maturity_btn = gr.Button(
            "Run Maturity",
            variant="primary",
        )

        maturity_image = gr.Image(
            type="pil",
            label="Detection / Result Image",
        )

        maturity_result = gr.Markdown()

        maturity_btn.click(
            fn=run_maturity,
            inputs=[
                fruit_image,
                fruit_state,
            ],
            outputs=[
                maturity_image,
                maturity_result,
            ],
        )

    with gr.Tab(
        "🌿 Leaf Disease"
    ):

        gr.Markdown(
            """
Upload a **second image containing the leaf**.

The disease model is selected automatically using
the fruit class from Step 1.
"""
        )

        leaf_image = gr.Image(
            type="pil",
            label="Upload Leaf Image",
        )

        disease_btn = gr.Button(
            "Run Disease Detection",
            variant="primary",
        )

        disease_result = gr.Markdown()

        disease_btn.click(
            fn=run_disease,
            inputs=[
                leaf_image,
                fruit_state,
            ],
            outputs=disease_result,
        )

    # =========================
    # RESET
    # =========================

    clear_btn = gr.Button("Clear")

    def clear_all():
        return (
            None,  # fruit image
            None,  # leaf image
            "",    # fruit state
            "Upload a fruit image and click **Classify Fruit**.",
            None,  # maturity image
            "",    # maturity result
            "",    # disease result
        )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            fruit_image,
            leaf_image,
            fruit_state,
            fruit_result,
            maturity_image,
            maturity_result,
            disease_result,
        ],
    )


if __name__ == "__main__":
    demo.launch()
