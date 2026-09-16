# Tomato Leaf Disease Classification — MobileNetV2

Image classification pipeline that identifies tomato leaf diseases from a photo and returns a treatment recommendation. Built with TensorFlow/Keras using transfer learning on MobileNetV2.

## Overview

- **Task**: Multi-class image classification of tomato leaf diseases (8 classes)
- **Backbone**: MobileNetV2 (ImageNet weights), fine-tuned in two stages
- **Input size**: 224×224 RGB
- **Extra**: A rule-based treatment recommendation engine maps each predicted class to severity, urgency, and organic/chemical treatment guidance

## Classes

| Index | Class |
|---|---|
| 0 | Tomato_Bacterial_spot |
| 1 | Tomato_Leaf_Mold |
| 2 | Tomato_Septoria_leaf_spot |
| 3 | Tomato_Spider_mites_Two_spotted_spider_mite |
| 4 | Tomato__Target_Spot |
| 5 | Tomato__Tomato_YellowLeaf__Curl_Virus |
| 6 | Tomato__Tomato_mosaic_virus |
| 7 | Tomato_healthy |

The mapping is saved to `class_indices.json` during preprocessing and reused at inference time.

## Dataset & Preprocessing

- Source folder: `final-tomato-dataset/grad project` (Kaggle dataset)
- Split into train/val/test (**70/15/15**) with `split-folders`, fixed seed (42)
- `ImageDataGenerator` for augmentation on the training set: rotation, width/height shift, zoom, horizontal flip, brightness jitter
- All splits preprocessed with MobileNetV2's `preprocess_input`

Resulting split sizes: 9,167 train / 1,961 val / 1,974 test images.

## Model

```
Input (224, 224, 3)
  → MobileNetV2 base (ImageNet weights)
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(128, relu)
  → Dropout(0.2)
  → Dense(num_classes, softmax)
```

## Training (two stages)

**Stage 1 — head only**
- Base frozen, classifier head trained from scratch
- Optimizer: Adam (lr = 1e-3), loss: categorical cross-entropy
- Up to 30 epochs, `EarlyStopping` (patience 7, restore best weights), `ModelCheckpoint` on `val_accuracy`
- Saved: `mobilenetv2_stage1_best.keras`, `mobilenetv2_stage1_final.keras`

**Stage 2 — fine-tuning**
- Last ~30 layers of the base unfrozen, rest stay frozen
- Optimizer: Adam (lr = 1e-5) — low LR to avoid destroying pretrained features
- Up to 20 epochs, same early stopping / checkpoint setup
- Saved: `mobilenetv2_finetuned_best.keras`, `mobilenetv2_finetuned_final.keras`

## Evaluation

- `classification_report` (precision/recall/F1 per class) on the test split
- Confusion matrix plotted with seaborn

## Inference

```python
model = load_trained_model("mobilenetv2_stage1_best.keras")
predicted_class, confidence = predict_image(model, image_path)
recommendation = get_recommendation(predicted_class, confidence)
```

`predict_image` resizes the image to 224×224, applies `preprocess_input`, and returns the top class + confidence.

Confidence thresholds used by `get_recommendation`:
- `< 0.60` → low confidence, asks for a clearer photo instead of a recommendation
- `0.60 – 0.85` → recommendation returned, flagged as moderate confidence
- `≥ 0.85` → recommendation returned, flagged as high confidence

An `ipywidgets` file uploader cell is included for interactive testing inside the notebook.

## Treatment Recommendation Engine

A small rule-based module (`TreatmentInfo` dataclass + `TREATMENT_DB`) that, for each disease class, returns:
- Severity (`low` / `medium` / `high`)
- Urgency note (how fast to act)
- Immediate actions
- Organic treatment options
- Chemical treatment options
- Prevention tips

> Output is general agricultural guidance based on common practice, not a substitute for a local agricultural extension officer/agronomist — worth surfacing that disclaimer in any app UI built on top of this.

## Requirements

```
tensorflow
split-folders
numpy
scikit-learn
seaborn
matplotlib
ipywidgets
```

## Project Structure (outputs)

```
class_indices.json                    # index -> class name mapping
mobilenetv2_stage1_best.keras         # best head-only checkpoint
mobilenetv2_stage1_final.keras        # final head-only model
mobilenetv2_finetuned_best.keras      # best fine-tuned checkpoint
mobilenetv2_finetuned_final.keras     # final fine-tuned model
```
