"""
FruitVision AI - Smart Precision Agriculture
Streamlit app that classifies a fruit image, then lets the user
check maturity or disease, loading all models from a local
models/ folder.
Pipeline:
1. User uploads a fruit image -> auto-classified as Apple/Tomato.
2. User picks a task: Maturity or Disease.
   - Maturity: classified immediately on the SAME uploaded image.
     - Apple: YOLO detects every apple in the image, then each
       detected apple is cropped and passed to the EfficientNet
       maturity model (which also takes the full annotated frame
       as a second input) to classify its ripeness.
     - Tomato: single YOLO classification model on the whole image.
   - Disease: user uploads a separate LEAF image, which is then
     routed to the apple or tomato disease model based on step 1's
     classification result. A full treatment recommendation
     (severity, urgency, immediate actions, organic/chemical
     treatment, prevention) is shown alongside the prediction.
"""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw

# Let TensorFlow use an available GPU without pre-allocating all GPU memory.
try:
    _gpus = tf.config.list_physical_devices("GPU")
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

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
# PATHS - all models live in ./models next to this file
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FRUIT_CLASSIFIER_PATH = MODELS_DIR / "fruit_classifier.keras"

# Apple maturity = YOLO detector + EfficientNet SavedModel (two-input pipeline)
APPLE_YOLO_PATH = MODELS_DIR / "apple_yolo11n_120_best.pt"
APPLE_MATURITY_PATH = MODELS_DIR / "apple_maturity_savedmodel"   # a FOLDER, not a .keras file

TOMATO_MATURITY_PATH = MODELS_DIR / "tomato_maturity_last" / "tomato_maturity_best (1).pt"   # YOLO detection (3-class)

APPLE_DISEASE_PATH = MODELS_DIR / "apple_disease.keras"
TOMATO_DISEASE_PATH = MODELS_DIR / "tomato_disease.keras"

# ============================================================
# MODEL CONFIGURATION
# ============================================================
# IMPORTANT: class order must match the order used during training.
FRUIT_CLASSES = [
    "Apple",
    "Tomato",
]

APPLE_MATURITY_CLASSES = [
    "unripe",
    "semi-ripe",
    "ripe",
]
APPLE_MATURITY_IMG_SIZE = 224
APPLE_YOLO_CONF = 0.25

TOMATO_MATURITY_CLASSES = [
    "unripe",
    "half_ripe",
    "ripe",
]
# From the model's deploy config.json (YOLOv8n detector, not a classifier)
TOMATO_YOLO_IMG_SIZE = 960
TOMATO_YOLO_CONF = 0.4
TOMATO_YOLO_IOU = 0.7

APPLE_DISEASE_CLASSES = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]

TOMATO_DISEASE_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# ============================================================
# TREATMENT RECOMMENDATIONS
# ============================================================
@dataclass
class TreatmentInfo:
    disease_name: str
    severity: str
    urgency: str
    immediate_actions: List[str] = field(default_factory=list)
    organic_treatment: List[str] = field(default_factory=list)
    chemical_treatment: List[str] = field(default_factory=list)
    prevention: List[str] = field(default_factory=list)

# ------------------------------------------------------------
# Apple disease treatment DB
# ------------------------------------------------------------
APPLE_TREATMENT_DB = {
    "complex": TreatmentInfo(
        disease_name="Complex (Multiple Diseases)",
        severity="high",
        urgency="Overlapping infections signal heavy disease pressure — inspect closely and act within 2-3 days",
        immediate_actions=[
            "Remove and destroy visibly infected leaves and fruit away from the orchard",
            "Prune out cankered or heavily spotted branches",
            "Inspect the leaf carefully to identify which diseases are present",
        ],
        organic_treatment=[
            "Apply a broad-spectrum copper or sulfur spray while the dominant disease is confirmed",
        ],
        chemical_treatment=[
            "Broad-spectrum protectant fungicide such as captan, then switch to a targeted product once the leading disease is identified",
        ],
        prevention=[
            "Improve airflow through regular pruning",
            "Maintain an orchard sanitation routine (remove mummified fruit and fallen leaves)",
            "Plant multi-disease-resistant apple varieties where possible",
        ],
    ),
    "frog_eye_leaf_spot": TreatmentInfo(
        disease_name="Frog Eye Leaf Spot",
        severity="medium",
        urgency="Spreads by rain-splashed spores starting at petal fall — act within a week",
        immediate_actions=[
            "Remove mummified fruit left on the tree or ground",
            "Prune out cankered branches and destroy them",
        ],
        organic_treatment=[
            "Copper-based fungicide sprays",
        ],
        chemical_treatment=[
            "Captan or thiophanate-methyl, applied from petal fall and repeated every 10-14 days in wet weather",
        ],
        prevention=[
            "Orchard sanitation — remove dead wood, mummies, and fallen fruit",
            "Avoid overhead irrigation to keep foliage dry",
            "Plant resistant varieties where possible",
        ],
    ),
    "healthy": TreatmentInfo(
        disease_name="Healthy",
        severity="low",
        urgency="No action required",
        immediate_actions=["Continue routine monitoring"],
        organic_treatment=[],
        chemical_treatment=[],
        prevention=["Maintain current watering, fertilizing, and pruning practices"],
    ),
    "powdery_mildew": TreatmentInfo(
        disease_name="Powdery Mildew",
        severity="medium",
        urgency="Spreads fastest in warm days with high humidity at night — act within 2-3 days",
        immediate_actions=[
            "Prune out visibly infected shoots and buds",
            "Reduce excess nitrogen fertilization, which favors new susceptible growth",
        ],
        organic_treatment=[
            "Sulfur-based fungicide sprays",
            "Neem oil as a lighter early-season option",
        ],
        chemical_treatment=[
            "Sterol-inhibiting fungicides such as myclobutanil, starting at bud swell and repeated every 7-10 days",
        ],
        prevention=[
            "Prune to improve air circulation around the canopy",
            "Plant mildew-resistant varieties where possible",
        ],
    ),
    "rust": TreatmentInfo(
        disease_name="Cedar-Apple Rust",
        severity="medium",
        urgency="Infects during wet spring weather when nearby juniper galls release spores — act at bud break",
        immediate_actions=[
            "Prune out infected leaves and branches",
            "Remove nearby juniper or cedar galls if feasible",
        ],
        organic_treatment=[
            "Limited organic options are effective once infection is established — prevention is key",
        ],
        chemical_treatment=[
            "Fungicides containing myclobutanil or propiconazole, starting at bud break and repeated every 7-14 days through spring",
        ],
        prevention=[
            "Plant rust-resistant apple varieties",
            "Increase distance from juniper/cedar trees where possible",
        ],
    ),
    "scab": TreatmentInfo(
        disease_name="Apple Scab",
        severity="high",
        urgency="Spreads rapidly in cool, wet spring weather starting at green tip — act promptly at season start",
        immediate_actions=[
            "Remove and destroy visibly infected leaves",
            "Rake and dispose of fallen leaves to cut next season's spore source",
        ],
        organic_treatment=[
            "Sulfur-based fungicide sprays",
        ],
        chemical_treatment=[
            "Captan or mancozeb, starting at bud break and repeated weekly in wet weather",
        ],
        prevention=[
            "Plant scab-resistant varieties",
            "Ensure adequate spacing and pruning for airflow",
            "Autumn sanitation of fallen leaves",
        ],
    ),
}

def get_apple_recommendation(
    predicted_class: str,
    confidence: float,
    medium_conf_threshold: float = 0.85,
) -> dict:
    if predicted_class not in APPLE_TREATMENT_DB:
        return {
            "status": "unknown_class",
            "message": "Class not found in the knowledge base — check the class name sent by the model.",
        }
    info = APPLE_TREATMENT_DB[predicted_class]
    if confidence >= medium_conf_threshold:
        confidence_note = "The model is highly confident."
    else:
        confidence_note = "Moderate confidence — visual confirmation is recommended before chemical treatment."
    return {
        "status": "ok",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "severity": info.severity,
        "disease_name": info.disease_name,
        "urgency": info.urgency,
        "immediate_actions": info.immediate_actions,
        "organic_treatment": info.organic_treatment,
        "chemical_treatment": info.chemical_treatment,
        "prevention": info.prevention,
    }

# ------------------------------------------------------------
# Tomato disease treatment DB
# ------------------------------------------------------------
TOMATO_TREATMENT_DB = {
    "Tomato_Bacterial_spot": TreatmentInfo(
        disease_name="Bacterial Spot",
        severity="medium",
        urgency="Spreads quickly in warm, humid conditions — act within 2 days",
        immediate_actions=[
            "Remove clearly infected leaves and destroy/dispose away from the field",
            "Reduce overhead watering; switch to drip irrigation if possible",
        ],
        organic_treatment=[
            "Apply copper-based bactericides on a regular schedule",
            "Improve airflow between plants to lower humidity",
        ],
        chemical_treatment=["Copper + mancozeb mix, following locally recommended dosage"],
        prevention=[
            "Crop rotation — avoid planting tomatoes in the same soil two years running",
            "Use certified disease-free seeds/seedlings",
        ],
    ),
    "Tomato_Leaf_Mold": TreatmentInfo(
        disease_name="Leaf Mold",
        severity="low",
        urgency="Relatively slow-spreading but weakens the crop — act within a week",
        immediate_actions=["Improve ventilation immediately", "Lower ambient humidity around the plant"],
        organic_treatment=["Sulfur-based fungicide spray"],
        chemical_treatment=["Chlorothalonil-based fungicide if needed"],
        prevention=["Good greenhouse ventilation", "Avoid overly dense planting"],
    ),
    "Tomato_Septoria_leaf_spot": TreatmentInfo(
        disease_name="Septoria Leaf Spot",
        severity="medium",
        urgency="Spreads from lower leaves upward — act within 3-4 days",
        immediate_actions=["Remove infected lower leaves immediately", "Clear away any infected plant debris"],
        organic_treatment=["Regular copper spray every 7-10 days during humid seasons"],
        chemical_treatment=["Mancozeb or chlorothalonil per local dosage guidance"],
        prevention=["At least 2-year crop rotation away from tomatoes", "Mulch to prevent soil-splash spread"],
    ),
    "Tomato_Spider_mites_Two_spotted_spider_mite": TreatmentInfo(
        disease_name="Two-Spotted Spider Mite",
        severity="medium",
        urgency="A pest, not a fungus/bacteria — spreads fast in hot, dry weather, act within 2-3 days",
        immediate_actions=["Check the undersides of leaves to confirm", "Spray plant with strong water jet to remove mites mechanically"],
        organic_treatment=["Neem oil or insecticidal soap", "Release natural predators (predatory mites) if feasible"],
        chemical_treatment=["Specialized miticides — regular insecticides usually don't work on mites"],
        prevention=["Maintain reasonable humidity", "Regularly inspect leaf undersides"],
    ),
    "Tomato__Target_Spot": TreatmentInfo(
        disease_name="Target Spot",
        severity="medium",
        urgency="Spreads in high humidity — act within 3-4 days",
        immediate_actions=["Remove visibly infected leaves/fruit", "Reduce how long leaves stay wet"],
        organic_treatment=["Regular copper spray"],
        chemical_treatment=["Fungicides with azoxystrobin or chlorothalonil"],
        prevention=["Good ventilation and adequate plant spacing", "Remove surrounding weeds"],
    ),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": TreatmentInfo(
        disease_name="Tomato Yellow Leaf Curl Virus",
        severity="high",
        urgency="Viral, transmitted by whiteflies — no direct cure, focus on stopping spread immediately",
        immediate_actions=["Remove the entire infected plant, roots included, immediately", "Dispose away from the rest of the crop"],
        organic_treatment=["No organic cure for the virus itself — focus on controlling the whitefly vector"],
        chemical_treatment=["Insecticides targeting whiteflies to prevent spread to other plants"],
        prevention=["Insect netting on greenhouses", "Virus-resistant varieties next planting", "Proactive whitefly control"],
    ),
    "Tomato__Tomato_mosaic_virus": TreatmentInfo(
        disease_name="Tomato Mosaic Virus",
        severity="high",
        urgency="Viral and highly contact-contagious — act immediately to prevent spread",
        immediate_actions=[
            "Remove the infected plant immediately, roots included",
            "Sterilize any tools that touched it before using on other plants",
            "Wash hands thoroughly after handling the infected plant",
        ],
        organic_treatment=["No cure for the virus itself — focus solely on preventing spread"],
        chemical_treatment=["No chemical cures the virus — focus on isolation and sanitation"],
        prevention=["Certified virus-free seeds", "Sterilize tools between plants", "Avoid smoking near plants (virus can transfer from tobacco)"],
    ),
    "Tomato_healthy": TreatmentInfo(
        disease_name="Healthy",
        severity="low",
        urgency="No action required",
        immediate_actions=["Continue routine monitoring"],
        organic_treatment=[],
        chemical_treatment=[],
        prevention=["Maintain current watering and fertilizing practices"],
    ),
}

def get_tomato_recommendation(
    predicted_class: str,
    confidence: float,
    low_conf_threshold: float = 0.6,
    medium_conf_threshold: float = 0.85,
) -> dict:
    if predicted_class not in TOMATO_TREATMENT_DB:
        return {
            "status": "unknown_class",
            "message": "Class not found in the knowledge base — check the class name sent by the model.",
        }
    info = TOMATO_TREATMENT_DB[predicted_class]
    if confidence < low_conf_threshold:
        return {
            "status": "low_confidence",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "message": (
                f"The model isn't confident enough (below {int(low_conf_threshold * 100)}%). "
                "Try a clearer photo or consult an expert before taking action."
            ),
        }
    if confidence >= medium_conf_threshold:
        confidence_note = "The model is highly confident."
    else:
        confidence_note = "Moderate confidence — visual confirmation is recommended before chemical treatment."
    return {
        "status": "ok",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "severity": info.severity,
        "disease_name": info.disease_name,
        "urgency": info.urgency,
        "immediate_actions": info.immediate_actions,
        "organic_treatment": info.organic_treatment,
        "chemical_treatment": info.chemical_treatment,
        "prevention": info.prevention,
    }

# ============================================================
# INPUT SIZES
# ============================================================
FRUIT_CLASSIFIER_SIZE = (224, 224)
APPLE_DISEASE_SIZE = (380, 380)
TOMATO_DISEASE_SIZE = (224, 224)

# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_raw(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """
    Resize image and return raw RGB pixels in [0, 255].
    Use this when the model has its own Rescaling layer
    or was trained directly on pixel values.
    """
    image = image.convert("RGB").resize(size)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)

def preprocess_01(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """
    Resize image and normalize pixels to [0, 1].
    Use this ONLY if the training notebook explicitly divided
    images by 255 before training.
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
# MODEL LOADING (cached so each model loads only once per session)
# ============================================================
@st.cache_resource
def load_keras_model(path_str: str):
    try:
        return tf.keras.models.load_model(path_str)
    except Exception as e:
        # Fallback for models saved with legacy/old Keras that tf.keras
        # (Keras 3) can't deserialize directly. This only kicks in when
        # the normal load fails, so it never affects models that load fine.
        try:
            import tf_keras
            return tf_keras.models.load_model(path_str)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load the model with both methods.\n"
                f"Primary error: {e}\n"
                f"tf_keras error: {fallback_error}"
            )

@st.cache_resource
def load_yolo_model(path_str: str):
    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )
    return YOLO(path_str)

@st.cache_resource
def load_apple_maturity_pipeline():
    """Loads the apple maturity YOLO detector + EfficientNet SavedModel together."""
    yolo = YOLO(str(APPLE_YOLO_PATH))
    saved_model = tf.saved_model.load(str(APPLE_MATURITY_PATH))
    infer = saved_model.signatures["serving_default"]
    return yolo, infer

# ============================================================
# OUTPUT INTERPRETATION
# ============================================================
def scores_to_prediction(raw_output: np.ndarray, class_names: list[str]) -> dict:
    """
    Convert a raw model output into class probabilities.
    Supports: single sigmoid output, 2-value binary output,
    softmax probabilities, and raw logits.
    """
    raw_output = np.asarray(raw_output).squeeze()
    if raw_output.ndim == 0:
        probability = float(raw_output)
        if not 0.0 <= probability <= 1.0:
            probability = float(tf.sigmoid(probability).numpy())
        if len(class_names) != 2:
            raise ValueError("Single output requires exactly 2 classes.")
        scores = np.array([1.0 - probability, probability], dtype=np.float32)
    elif raw_output.shape == (1,) and len(class_names) == 2:
        probability = float(raw_output[0])
        if not 0.0 <= probability <= 1.0:
            probability = float(tf.sigmoid(probability).numpy())
        scores = np.array([1.0 - probability, probability], dtype=np.float32)
    else:
        scores = raw_output.astype(np.float32).flatten()
        if len(scores) != len(class_names):
            raise ValueError(
                f"Model output has {len(scores)} values, "
                f"but {len(class_names)} classes were configured."
            )
        if np.any(scores < 0) or not np.isclose(np.sum(scores), 1.0, atol=1e-3):
            scores = tf.nn.softmax(scores).numpy()
        scores = scores / np.sum(scores)
    best_index = int(np.argmax(scores))
    return {
        "label": class_names[best_index],
        "confidence": float(scores[best_index]),
        "scores": list(zip(class_names, scores.tolist())),
    }

# ============================================================
# KERAS PREDICTION
# ============================================================
def predict_keras_classifier(
    image: Image.Image,
    model_path: Path,
    class_names: list[str],
    input_size: tuple[int, int],
    preprocess_method: str = "raw",
) -> dict:
    if not model_path.exists():
        return {"ready": False, "error": f"Model file not found:\n{model_path}"}
    try:
        model = load_keras_model(str(model_path))
        batch = preprocess_for(image, input_size, preprocess_method)
        raw_output = model.predict(batch, verbose=0)[0]
        result = scores_to_prediction(raw_output, class_names)
        result["ready"] = True
        result["model_path"] = str(model_path)
        return result
    except Exception as error:
        return {"ready": False, "error": str(error), "model_path": str(model_path)}

# ============================================================
# YOLO PREDICTION (classification-mode YOLO, e.g. tomato maturity)
# ============================================================
def predict_yolo(
    image: Image.Image,
    model_path: Path,
    class_names: Optional[list[str]] = None,
) -> dict:
    if not model_path.exists():
        return {"ready": False, "error": f"Model file not found:\n{model_path}"}
    if YOLO is None:
        return {
            "ready": False,
            "error": "ultralytics is not installed.\nRun: pip install ultralytics",
        }
    try:
        model = load_yolo_model(str(model_path))
        results = model.predict(source=np.asarray(image.convert("RGB")), verbose=False)
        if not results:
            return {"ready": False, "error": "YOLO returned no results."}
        result = results[0]
        # Classification-mode YOLO
        if result.probs is not None:
            probabilities = result.probs.data.cpu().numpy()
            names = result.names
            if class_names is None:
                class_names = [names[i] for i in range(len(probabilities))]
            if len(class_names) != len(probabilities):
                return {
                    "ready": False,
                    "error": (
                        f"YOLO returned {len(probabilities)} classes, "
                        f"but {len(class_names)} were configured."
                    ),
                }
            prediction = scores_to_prediction(probabilities, class_names)
            prediction["ready"] = True
            prediction["model_path"] = str(model_path)
            prediction["model_type"] = "classification"
            return prediction
        # Detection-mode YOLO (kept in case you swap in a detection model)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            best_index = int(np.argmax(boxes.conf.cpu().numpy()))
            confidence = float(boxes.conf[best_index].cpu().item())
            class_id = int(boxes.cls[best_index].cpu().item())
            label = result.names.get(class_id, str(class_id))
            detected_objects = []
            for i in range(len(boxes)):
                detected_class_id = int(boxes.cls[i].cpu().item())
                detected_objects.append({
                    "label": result.names.get(detected_class_id, str(detected_class_id)),
                    "confidence": float(boxes.conf[i].cpu().item()),
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
        return {"ready": False, "error": "No prediction produced by YOLO model."}
    except Exception as error:
        return {"ready": False, "error": str(error), "model_path": str(model_path)}

# ============================================================
# APPLE MATURITY MULTI-DETECTION (YOLO -> crop each apple -> EfficientNet)
# ============================================================
def _prepare_apple_crop(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((APPLE_MATURITY_IMG_SIZE, APPLE_MATURITY_IMG_SIZE))
    return np.asarray(image, dtype=np.float32)

def prepare_image(image: Image.Image) -> np.ndarray:
    """Prepare an image exactly like the Test notebook for EfficientNet."""
    image = image.convert("RGB")
    image = image.resize((APPLE_MATURITY_IMG_SIZE, APPLE_MATURITY_IMG_SIZE))
    return np.asarray(image, dtype=np.float32)

def predict_apple_maturity_multi(image: Image.Image) -> dict:
    """
    Exact apple maturity inference flow used in the test notebook:
    1. YOLO is used ONLY as an apple detector.
       It produces bounding boxes + detection confidence.
       It does NOT assign maturity labels and does not classify apples.
    2. Every YOLO bounding box is cropped from the original image.
    3. EfficientNet receives TWO inputs for every crop:
       - apple_input: cropped apple
       - tree_input: the FULL annotated tree/frame containing all YOLO boxes
    4. EfficientNet predicts maturity for each detected apple independently.
    """
    if not APPLE_YOLO_PATH.exists():
        return {"ready": False, "error": f"YOLO model file not found:\n{APPLE_YOLO_PATH}"}
    if not APPLE_MATURITY_PATH.exists():
        return {"ready": False, "error": f"Maturity SavedModel folder not found:\n{APPLE_MATURITY_PATH}"}
    if YOLO is None:
        return {
            "ready": False,
            "error": "ultralytics is not installed.\nRun: pip install ultralytics",
        }
    try:
        yolo_model, infer = load_apple_maturity_pipeline()
        # YOLO = DETECTOR ONLY.
        # No class/maturity labels are taken from YOLO.
        # Same YOLO invocation as the Test notebook.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            yolo_source_path = tmp.name
        image.convert("RGB").save(yolo_source_path, format="JPEG")
        try:
            results = yolo_model.predict(
                source=[yolo_source_path],
                imgsz=640,
                conf=APPLE_YOLO_CONF,
                save=False,
                verbose=False
            )
        finally:
            Path(yolo_source_path).unlink(missing_ok=True)

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return {
                "ready": True,
                "detections": [],
                "message": "YOLO did not detect any apples in the image.",
            }

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()

        boxed_tree = image.copy()
        draw = ImageDraw.Draw(boxed_tree)

        apple_batch = []
        tree_batch = []
        box_data = []

        # Same loop/order as the Test notebook.
        for box, score in zip(xyxy, confs):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, image.width - 1))
            y1 = max(0, min(y1, image.height - 1))
            x2 = max(0, min(x2, image.width))
            y2 = max(0, min(y2, image.height))
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            apple_crop = image.crop((x1, y1, x2, y2))
            apple_batch.append(prepare_image(apple_crop))
            tree_batch.append(prepare_image(boxed_tree))
            box_data.append({
                "box": [x1, y1, x2, y2],
                "yolo_confidence": float(score),
            })

        if not apple_batch:
            return {
                "ready": True,
                "detections": [],
                "message": "No valid boxes remaining after filtering.",
            }

        apple_batch = np.asarray(apple_batch, dtype=np.float32)
        tree_batch = np.asarray(tree_batch, dtype=np.float32)

        # EfficientNet maturity classification.
        # IMPORTANT: do NOT send all detected apples in one huge batch.
        # A crowded tree can contain 100+ boxes, which can exhaust CPU RAM.
        # We still process EVERY box, but in small batches.
        INFERENCE_BATCH_SIZE = 8
        probability_batches = []
        for start in range(0, len(apple_batch), INFERENCE_BATCH_SIZE):
            end = start + INFERENCE_BATCH_SIZE
            apple_chunk = tf.convert_to_tensor(apple_batch[start:end])
            tree_chunk = tf.convert_to_tensor(tree_batch[start:end])
            chunk_output = infer(
                apple_input=apple_chunk,
                tree_input=tree_chunk,
            )
            chunk_probabilities = list(chunk_output.values())[0].numpy()
            probability_batches.append(chunk_probabilities)

        probabilities = np.concatenate(probability_batches, axis=0)

        predictions = []
        for i, probs in enumerate(probabilities):
            class_id = int(np.argmax(probs))
            predictions.append({
                "box": box_data[i]["box"],
                "yolo_confidence": box_data[i]["yolo_confidence"],
                "maturity": APPLE_MATURITY_CLASSES[class_id],
                "maturity_confidence": float(probs[class_id]),
                "probabilities": {
                    APPLE_MATURITY_CLASSES[j]: float(probs[j])
                    for j in range(len(APPLE_MATURITY_CLASSES))
                },
            })

        return {
            "ready": True,
            "detections": predictions,
            "boxed_tree": boxed_tree,
        }
    except Exception as error:
        return {"ready": False, "error": str(error)}

def show_apple_maturity_multi(result: dict, original_image: Image.Image):
    """
    Streamlit presentation matching the test notebook:
    - YOLO Bounding Boxes: boxes only, no maturity labels/counts.
    - Apple Maturity Prediction: EfficientNet maturity labels on the boxes.
    - Maturity summary: total count of ripe / semi-ripe / unripe apples.
    - Per-box details are shown below.
    """
    if not result.get("ready"):
        st.error(result.get("error", "Unknown error."))
        return

    detections = result.get("detections", [])
    if not detections:
        st.info(result.get("message", "No apples visible in the image."))
        return

    # ============================================================
    # 1) YOLO BOUNDING BOXES
    # Detector output only: bounding boxes + YOLO confidence.
    # No maturity labels and no maturity/count summary.
    # ============================================================
    boxed_tree = result.get("boxed_tree", original_image.copy())
    st.markdown("### YOLO Bounding Boxes")
    st.image(
        boxed_tree,
        caption="YOLO detections",
        use_container_width=True,
    )

    # ============================================================
    # 2) APPLE MATURITY PREDICTION
    # EfficientNet predicts maturity for each YOLO crop.
    # ============================================================
    st.markdown("### Apple Maturity Prediction")
    result_image = original_image.copy()
    draw = ImageDraw.Draw(result_image)
    for pred in detections:
        x1, y1, x2, y2 = pred["box"]
        label = (
            f"{pred['maturity']} "
            f"{pred['maturity_confidence']:.2f}"
        )
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4,
        )
        draw.text(
            (x1, max(0, y1 - 18)),
            label,
            fill="red",
        )
    st.image(
        result_image,
        caption="Apple Maturity Prediction",
        use_container_width=True,
    )

    # ============================================================
    # 3) MATURITY SUMMARY — total count per class (unripe / semi-ripe / ripe)
    # ============================================================
    counts = Counter(d["maturity"] for d in detections)
    st.markdown("### Maturity Summary")
    st.write(f"Total apples detected: **{len(detections)}**")
    cols = st.columns(len(APPLE_MATURITY_CLASSES))
    for col, cls in zip(cols, APPLE_MATURITY_CLASSES):
        col.metric(cls, counts.get(cls, 0))

    # ============================================================
    # 4) DETAILS — equivalent to the notebook print output
    # ============================================================
    with st.expander(f"Details for each apple ({len(detections)})"):
        for i, pred in enumerate(detections, 1):
            st.markdown(f"**Box {i}**")
            st.write("Bounding box:", pred["box"])
            st.write(
                "YOLO confidence:",
                round(pred["yolo_confidence"], 4),
            )
            st.write("Maturity:", pred["maturity"])
            st.write(
                "Maturity confidence:",
                round(pred["maturity_confidence"], 4),
            )
            st.write(
                "Probabilities:",
                {
                    k: round(v, 4)
                    for k, v in pred["probabilities"].items()
                },
            )

# ============================================================
# TOMATO MATURITY MULTI-DETECTION (single-stage YOLO detector)
# ============================================================
def predict_tomato_maturity_multi(image: Image.Image) -> dict:
    """
    Runs the tomato YOLO detector once and returns every detected
    tomato with its box, class (unripe/half_ripe/ripe) and confidence.
    Unlike the apple pipeline this model classifies ripeness directly
    per-box, so there's no separate crop + EfficientNet step.
    """
    if not TOMATO_MATURITY_PATH.exists():
        return {"ready": False, "error": f"Model file not found:\n{TOMATO_MATURITY_PATH}"}
    if YOLO is None:
        return {"ready": False, "error": "ultralytics is not installed.\nRun: pip install ultralytics"}
    try:
        model = load_yolo_model(str(TOMATO_MATURITY_PATH))
        results = model.predict(
            source=np.asarray(image.convert("RGB")),
            imgsz=TOMATO_YOLO_IMG_SIZE,
            conf=TOMATO_YOLO_CONF,
            iou=TOMATO_YOLO_IOU,
            save=False,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return {"ready": True, "detections": [], "message": "YOLO did not detect any tomatoes in the image."}

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        cls_ids = result.boxes.cls.detach().cpu().numpy()

        detections = []
        for box, score, cls_id in zip(xyxy, confs, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, image.width - 1))
            y1 = max(0, min(y1, image.height - 1))
            x2 = max(0, min(x2, image.width))
            y2 = max(0, min(y2, image.height))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(cls_id)
            label = result.names.get(class_id, TOMATO_MATURITY_CLASSES[class_id] if class_id < len(TOMATO_MATURITY_CLASSES) else str(class_id))
            detections.append({
                "box": [x1, y1, x2, y2],
                "maturity": label,
                "maturity_confidence": float(score),
            })

        if not detections:
            return {"ready": True, "detections": [], "message": "No valid boxes remaining after filtering."}

        return {"ready": True, "detections": detections}
    except Exception as error:
        return {"ready": False, "error": str(error)}

def show_tomato_maturity_multi(result: dict, original_image: Image.Image):
    if not result.get("ready"):
        st.error(result.get("error", "Unknown error."))
        return

    detections = result.get("detections", [])
    if not detections:
        st.info(result.get("message", "No tomatoes visible in the image."))
        return

    # Draw all detected tomatoes on one annotated image
    annotated = original_image.copy()
    draw = ImageDraw.Draw(annotated)
    box_color = {"unripe": "green", "half_ripe": "orange", "ripe": "red"}
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = box_color.get(det["maturity"], "blue")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    st.image(annotated, caption=f"Detected {len(detections)} tomato(es)", use_container_width=True)

    # Summary counts per maturity class
    counts = Counter(d["maturity"] for d in detections)
    st.markdown("### Maturity Summary")
    cols = st.columns(len(TOMATO_MATURITY_CLASSES))
    for col, cls in zip(cols, TOMATO_MATURITY_CLASSES):
        col.metric(cls, counts.get(cls, 0))

    with st.expander(f"Details for each tomato ({len(detections)})"):
        for i, det in enumerate(detections, 1):
            st.write(
                f"**{i}.** {det['maturity']} "
                f"({det['maturity_confidence'] * 100:.0f}%)"
            )

    st.caption(
        "Note: This model's recall is approximately 0.48–0.49 on closely packed / clustered tomatoes, "
        "so it may miss some fruits in crowded images."
    )

# ============================================================
# RESULT DISPLAY HELPERS
# ============================================================
SEVERITY_STYLE = {
    "high": ("🔴", "error"),
    "medium": ("🟠", "warning"),
    "low": ("🟢", "success"),
}

def show_recommendation(recommendation: dict):
    """Render the full recommendation dict from get_apple_recommendation /
    get_tomato_recommendation underneath the prediction result."""
    status = recommendation.get("status")
    if status == "unknown_class":
        st.error(recommendation.get("message", "Unknown class."))
        return
    if status == "low_confidence":
        st.warning(recommendation.get("message", "Confidence too low."))
        return
    if status != "ok":
        return

    st.markdown("#### 🩺 Treatment recommendation")
    emoji, box_type = SEVERITY_STYLE.get(recommendation["severity"], ("⚪", "info"))
    header = f"{emoji} **{recommendation['disease_name']}** — severity: {recommendation['severity']}"
    getattr(st, box_type)(header)
    st.caption(f"{recommendation['confidence_note']}  ·  {recommendation['urgency']}")

    col1, col2 = st.columns(2)
    with col1:
        if recommendation["immediate_actions"]:
            st.markdown("**⚡ Immediate actions**")
            for item in recommendation["immediate_actions"]:
                st.markdown(f"- {item}")
        if recommendation["organic_treatment"]:
            st.markdown("**🌿 Organic treatment**")
            for item in recommendation["organic_treatment"]:
                st.markdown(f"- {item}")
    with col2:
        if recommendation["chemical_treatment"]:
            st.markdown("**🧪 Chemical treatment**")
            for item in recommendation["chemical_treatment"]:
                st.markdown(f"- {item}")
        if recommendation["prevention"]:
            st.markdown("**🛡️ Prevention**")
            for item in recommendation["prevention"]:
                st.markdown(f"- {item}")

    st.caption(
        "General agricultural guidance based on common practice — not a substitute "
        "for a local agricultural extension officer or agronomist."
    )

def show_result(result: dict, fruit_type: Optional[str] = None):
    if not result.get("ready"):
        st.error(result.get("error", "Unknown error."))
        return
    st.success(f"**{result['label']}** — {result['confidence'] * 100:.1f}% confidence")
    scores = dict(result.get("scores", []))
    if scores:
        st.bar_chart(scores)
    if fruit_type == "Apple":
        recommendation = get_apple_recommendation(result["label"], result["confidence"])
        show_recommendation(recommendation)
    elif fruit_type == "Tomato":
        recommendation = get_tomato_recommendation(result["label"], result["confidence"])
        show_recommendation(recommendation)

# ============================================================
# SIDEBAR - model status
# ============================================================
with st.sidebar:
    st.header("Model status")
    model_checks = [
        ("Fruit classifier", FRUIT_CLASSIFIER_PATH),
        ("Apple maturity - YOLO detector", APPLE_YOLO_PATH),
        ("Apple maturity - EfficientNet", APPLE_MATURITY_PATH),
        ("Tomato maturity (YOLO)", TOMATO_MATURITY_PATH),
        ("Apple disease", APPLE_DISEASE_PATH),
        ("Tomato disease", TOMATO_DISEASE_PATH),
    ]
    for name, path in model_checks:
        if path.exists():
            st.success(f"✅ {name}")
        else:
            st.warning(f"⚠️ {name} — not found:\n`{path.relative_to(BASE_DIR)}`")
    if st.button("🔄 Start over"):
        st.session_state.clear()
        st.rerun()

# ============================================================
# SESSION STATE
# ============================================================
if "task" not in st.session_state:
    st.session_state.task = None

# ============================================================
# MAIN PAGE
# ============================================================
st.title("🍎 FruitVision AI")
st.subheader("Smart Precision Agriculture")
st.write("Upload a fruit photo to identify it, then check its maturity or screen it for disease.")

# ------------------------------------------------------------
# STEP 1 - fruit image upload + auto classification
# ------------------------------------------------------------
st.markdown("## 1. Upload a fruit image")
fruit_image_file = st.file_uploader(
    "Upload a photo of the fruit (apple or tomato)",
    type=["jpg", "jpeg", "png", "webp"],
    key="fruit_upload",
)
if fruit_image_file is None:
    st.stop()

fruit_image = Image.open(fruit_image_file).convert("RGB")
st.image(fruit_image, caption="Uploaded fruit", use_container_width=True)

fruit_result = predict_keras_classifier(
    image=fruit_image,
    model_path=FRUIT_CLASSIFIER_PATH,
    class_names=FRUIT_CLASSES,
    input_size=FRUIT_CLASSIFIER_SIZE,
    preprocess_method="raw",
)
if not fruit_result.get("ready"):
    st.error(fruit_result.get("error"))
    st.stop()

fruit_type = fruit_result["label"]
st.success(
    f"Classified as **{fruit_type}** "
    f"({fruit_result['confidence'] * 100:.1f}% confidence)"
)

# ------------------------------------------------------------
# STEP 2 - task selection (Maturity / Disease)
# ------------------------------------------------------------
st.markdown("## 2. Choose what to check")
col1, col2 = st.columns(2)
with col1:
    if st.button("🍏 Maturity", use_container_width=True):
        st.session_state.task = "maturity"
with col2:
    if st.button("🍃 Disease", use_container_width=True):
        st.session_state.task = "disease"

# ------------------------------------------------------------
# STEP 3a - MATURITY: runs immediately on the same fruit image
# ------------------------------------------------------------
if st.session_state.task == "maturity":
    st.markdown("## Maturity result")
    if fruit_type == "Apple":
        result = predict_apple_maturity_multi(fruit_image)
        show_apple_maturity_multi(result, fruit_image)
    else:
        result = predict_tomato_maturity_multi(fruit_image)
        show_tomato_maturity_multi(result, fruit_image)

# ------------------------------------------------------------
# STEP 3b - DISEASE: asks for a separate leaf image, then
# routes to the apple/tomato disease model based on STEP 1,
# and shows the matching treatment recommendation.
# ------------------------------------------------------------
elif st.session_state.task == "disease":
    st.markdown("## Disease screening")
    st.write(f"Upload a **leaf** photo — it will be checked with the {fruit_type} disease model.")
    leaf_image_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        key="leaf_upload",
    )
    if leaf_image_file is not None:
        leaf_image = Image.open(leaf_image_file).convert("RGB")
        st.image(leaf_image, caption="Uploaded leaf", use_container_width=True)
        if fruit_type == "Apple":
            result = predict_keras_classifier(
                image=leaf_image,
                model_path=APPLE_DISEASE_PATH,
                class_names=APPLE_DISEASE_CLASSES,
                input_size=APPLE_DISEASE_SIZE,
                preprocess_method="raw",
            )
        else:
            result = predict_keras_classifier(
                image=leaf_image,
                model_path=TOMATO_DISEASE_PATH,
                class_names=TOMATO_DISEASE_CLASSES,
                input_size=TOMATO_DISEASE_SIZE,
                preprocess_method="raw",
            )
        show_result(result, fruit_type=fruit_type)