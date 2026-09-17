import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from functools import lru_cache
from pathlib import Path
import spaces

# ============================================================
# Smart Precision Agriculture - Hugging Face ZeroGPU
# ============================================================
# TensorFlow and Ultralytics are intentionally NOT imported at
# startup. They are imported lazily inside the model-loading
# functions so the Space can start safely on ZeroGPU.
# ============================================================

ROOT = Path(__file__).resolve().parent

CLASSIFIER_NAME = "apple_tomato_efficientnetb0_final2.keras"

APPLE_DETECTOR_NAME = "apple_yolo11n_120_best.pt"
APPLE_MATURITY_NAME = "apple_maturity_model.keras"

TOMATO_MATURITY_NAME = "tomato_maturity_best (1).pt"

APPLE_DISEASE_NAME = "best_model FF.keras"
TOMATO_DISEASE_NAME = "mobilenetv2_stage1_best.keras"

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
    matches = list(ROOT.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Model file was not found in the Space repository: {filename}"
        )
    return matches[0]


def pil_rgb(image):
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    return Image.fromarray(np.asarray(image)).convert("RGB")


def preprocess_for_shape(image, shape):
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
    import tensorflow as tf

    return tf.keras.models.load_model(
        find_model(CLASSIFIER_NAME),
        compile=False,
    )


@lru_cache(maxsize=1)
def load_apple_detector():
    from ultralytics import YOLO

    return YOLO(str(find_model(APPLE_DETECTOR_NAME)))


@lru_cache(maxsize=1)
def load_apple_maturity():
    import tensorflow as tf

    return tf.keras.models.load_model(
        find_model(APPLE_MATURITY_NAME),
        compile=False,
    )


@lru_cache(maxsize=1)
def load_tomato_maturity():
    from ultralytics import YOLO

    return YOLO(str(find_model(TOMATO_MATURITY_NAME)))


@lru_cache(maxsize=1)
def load_apple_disease():
    import tensorflow as tf

    return tf.keras.models.load_model(
        find_model(APPLE_DISEASE_NAME),
        compile=False,
    )


@lru_cache(maxsize=1)
def load_tomato_disease():
    import tensorflow as tf

    return tf.keras.models.load_model(
        find_model(TOMATO_DISEASE_NAME),
        compile=False,
    )


# ------------------------------------------------------------
# 1. Fruit classification
# ------------------------------------------------------------

def classifier_predict(image):
    model = load_classifier()

    x = preprocess_for_shape(
        image,
        model.inputs[0].shape,
    )

    y = np.asarray(
        model.predict(x, verbose=0)
    )

    if y.ndim == 1:
        y = y.reshape(1, -1)

    # Binary sigmoid:
    # output = Tomato probability
    if y.shape[-1] == 1:
        tomato_probability = float(y.reshape(-1)[0])

        # Safety if the model returns a logit instead of probability.
        if tomato_probability < 0.0 or tomato_probability > 1.0:
            tomato_probability = 1.0 / (
                1.0 + np.exp(-tomato_probability)
            )

        apple_probability = 1.0 - tomato_probability

        if tomato_probability >= 0.5:
            return "Tomato", tomato_probability

        return "Apple", apple_probability

    # Two-class softmax fallback.
    idx = int(np.argmax(y[0]))

    names = getattr(model, "class_names", None)
    names = list(names) if names else ["Apple", "Tomato"]

    label = (
        str(names[idx])
        if idx < len(names)
        else str(idx)
    )

    return label, float(y[0][idx])


@spaces.GPU
def run_classification(image):
    if image is None:
        return "", "Please upload an apple or tomato image first."

    try:
        label, confidence = classifier_predict(image)

        result = (
            "### 🍎🍅 Fruit Classification\n\n"
            f"**{label}**\n\n"
            f"Confidence: **{confidence:.2%}**"
        )

        return label, result

    except Exception as e:
        return "", (
            "### ❌ Classification Error\n"
            f"`{type(e).__name__}: {e}`"
        )


# ------------------------------------------------------------
# 2. Apple YOLO detection
# ------------------------------------------------------------

def apple_detection_and_crops(image):
    detector = load_apple_detector()
    original = pil_rgb(image)

    result = detector(
        original,
        conf=0.25,
        verbose=False,
    )[0]

    boxes = []
    crops = []

    if result.boxes is None or len(result.boxes) == 0:
        return boxes, crops

    for box in result.boxes:
        xyxy = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        conf = float(
            box.conf[0]
            .detach()
            .cpu()
            .item()
        )

        x1, y1, x2, y2 = [
            int(round(v))
            for v in xyxy
        ]

        x1 = max(0, min(x1, original.width))
        y1 = max(0, min(y1, original.height))
        x2 = max(0, min(x2, original.width))
        y2 = max(0, min(y2, original.height))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append((x1, y1, x2, y2, conf))
        crops.append(original.crop((x1, y1, x2, y2)))

    return boxes, crops


def annotate_boxes(image, boxes):
    out = pil_rgb(image).copy()
    draw = ImageDraw.Draw(out)

    for i, (x1, y1, x2, y2, conf) in enumerate(
        boxes,
        start=1,
    ):
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
#    YOLO crop + original image -> 2-input EfficientNet
# ------------------------------------------------------------

def apple_maturity_predict(original_image, crop_image):
    model = load_apple_maturity()

    if len(model.inputs) != 2:
        raise ValueError(
            "Apple maturity model must have exactly 2 inputs, "
            f"but found {len(model.inputs)}."
        )

    shapes = [tuple(t.shape) for t in model.inputs]
    names = [str(t.name).lower() for t in model.inputs]

    crop_idx = next(
        (
            i for i, name in enumerate(names)
            if "crop" in name
        ),
        None,
    )

    original_idx = next(
        (
            i for i, name in enumerate(names)
            if "original" in name or "full" in name
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
        # input 0 = YOLO crop
        # input 1 = original classifier image
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

    if prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError(
            "Unexpected Apple maturity output shape: "
            f"{prediction.shape}. Expected 3 classes: "
            "Unripe, Semi-ripe, Ripe."
        )

    idx = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][idx])

    return APPLE_MATURITY_CLASSES[idx], confidence


# ------------------------------------------------------------
# 4. Tomato maturity
# ------------------------------------------------------------

def tomato_maturity_predict(image):
    model = load_tomato_maturity()

    result = model(
        pil_rgb(image),
        verbose=False,
    )[0]

    # Classification model.
    if result.probs is not None:
        idx = int(result.probs.top1)
        confidence = float(result.probs.top1conf)

        names = result.names

        if isinstance(names, dict):
            label = names.get(idx, str(idx))
        else:
            label = str(idx)

        return str(label), confidence, None

    # Detection fallback.
    if result.boxes is not None and len(result.boxes) > 0:
        annotated = result.plot(
            labels=True,
            conf=True,
        )

        annotated = Image.fromarray(annotated[..., ::-1])

        best_idx = int(
            np.argmax(
                result.boxes.conf.detach().cpu().numpy()
            )
        )

        best_conf = float(
            result.boxes.conf[best_idx]
            .detach()
            .cpu()
            .item()
        )

        cls_idx = int(
            result.boxes.cls[best_idx]
            .detach()
            .cpu()
            .item()
        )

        names = result.names

        if isinstance(names, dict):
            label = names.get(cls_idx, str(cls_idx))
        else:
            label = str(cls_idx)

        return str(label), best_conf, annotated

    raise ValueError(
        "The Tomato maturity .pt model returned neither "
        "classification probabilities nor detection boxes."
    )


# ------------------------------------------------------------
# 5. Maturity workflow
# ------------------------------------------------------------

@spaces.GPU
def run_maturity(fruit_image, fruit):
    if fruit_image is None:
        return None, "Please upload the fruit image first."

    if not fruit:
        return None, "Please click **Classify Fruit** first."

    try:
        if fruit.lower() == "apple":
            boxes, crops = apple_detection_and_crops(
                fruit_image
            )

            if not crops:
                return (
                    fruit_image,
                    "### 🍎 Apple Maturity\n\n"
                    "❌ No apple was detected by the Apple YOLO detector.",
                )

            annotated = annotate_boxes(
                fruit_image,
                boxes,
            )

            results = []

            for crop in crops:
                label, confidence = apple_maturity_predict(
                    fruit_image,
                    crop,
                )
                results.append((label, confidence))

            summary = "### 🍎 Apple Maturity\n\n"

            for i, (label, confidence) in enumerate(
                results,
                start=1,
            ):
                summary += (
                    f"**Apple {i}:** {label} "
                    f"({confidence:.2%})\n\n"
                )

            return annotated, summary

        label, confidence, annotated = tomato_maturity_predict(
            fruit_image
        )

        output_image = (
            annotated
            if annotated is not None
            else fruit_image
        )

        return (
            output_image,
            "### 🍅 Tomato Maturity\n\n"
            f"**{label}** — {confidence:.2%}",
        )

    except Exception as e:
        return (
            fruit_image,
            "### ❌ Maturity Error\n"
            f"`{type(e).__name__}: {e}`",
        )


# ------------------------------------------------------------
# 6. Leaf disease
# ------------------------------------------------------------

def keras_classifier_predict(model, image, class_names):
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

    if prediction.size == 1:
        probability = float(
            np.ravel(prediction)[0]
        )

        if probability < 0.0 or probability > 1.0:
            probability = 1.0 / (
                1.0 + np.exp(-probability)
            )

        idx = 1 if probability >= 0.5 else 0

        if idx >= len(class_names):
            raise ValueError(
                f"Binary disease model needs 2 class names, "
                f"but only {len(class_names)} were supplied."
            )

        label = class_names[idx]
        confidence = (
            probability
            if idx == 1
            else 1.0 - probability
        )

        return label, confidence

    idx = int(np.argmax(prediction))

    label = (
        class_names[idx]
        if idx < len(class_names)
        else f"Class {idx}"
    )

    return label, float(prediction[idx])


@spaces.GPU
def run_disease(leaf_image, fruit):
    if leaf_image is None:
        return "Please upload a leaf image first."

    if not fruit:
        return (
            "Please click **Classify Fruit** first "
            "so the correct disease model can be selected."
        )

    try:
        if fruit.lower() == "apple":
            model = load_apple_disease()

            label, confidence = keras_classifier_predict(
                model,
                leaf_image,
                APPLE_DISEASE_CLASSES,
            )
        else:
            model = load_tomato_disease()

            label, confidence = keras_classifier_predict(
                model,
                leaf_image,
                TOMATO_DISEASE_CLASSES,
            )

        return (
            f"### 🌿 {fruit} Leaf Disease\n\n"
            f"**{label}** — {confidence:.2%}"
        )

    except Exception as e:
        return (
            "### ❌ Disease Error\n"
            f"`{type(e).__name__}: {e}`"
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

Upload an **Apple or Tomato** image.

1. The Fruit Classifier identifies **Apple / Tomato**.
2. Choose **Maturity** or **Leaf Disease**.
3. The correct model is selected automatically.
"""
    )

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
        fn=run_classification,
        inputs=fruit_image,
        outputs=[
            fruit_state,
            fruit_result,
        ],
    )

    gr.Markdown("## 2️⃣ Choose the Task")

    with gr.Tab("🍎🍅 Maturity"):
        gr.Markdown(
            """
**Apple:** Apple YOLO detects/crops the fruit → the crop and
the original fruit image are sent to the two-input EfficientNet
Apple maturity model.

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

    with gr.Tab("🌿 Leaf Disease"):
        gr.Markdown(
            """
Upload a **second image containing the leaf**.

The disease model is selected automatically from the
Apple/Tomato classification result.
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

    clear_btn = gr.Button("Clear")

    def clear_all():
        return (
            None,
            None,
            "",
            "Upload a fruit image and click **Classify Fruit**.",
            None,
            "",
            "",
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
