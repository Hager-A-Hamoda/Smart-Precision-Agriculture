# Smart Precision Agriculture

## About the Project

Smart Precision Agriculture is a computer vision-based system designed to analyze apple and tomato plants using deep learning models.

The system identifies whether the detected fruit is an apple or tomato, estimates fruit maturity, and detects diseases affecting the leaves.

The project combines object detection and image classification models and provides a Streamlit interface for deployment.

## Project Objectives

* Detect and classify apples and tomatoes.
* Estimate the maturity level of apples and tomatoes.
* Detect diseases affecting apple leaves.
* Detect diseases affecting tomato leaves.
* Integrate the trained models into a Streamlit application.

## System Overview

The project consists of five main computer vision tasks:

### 1. Apple & Tomato Detection and Classification

This part detects the fruit and identifies whether it is an apple or tomato.

* **YOLO** is used for object detection.
* **MobileNet** is used for classification.

### 2. Apple Maturity

This model classifies apples into three maturity levels:

* Unripe
* Semi-ripe
* Ripe

**EfficientNet** is used for apple maturity classification, while **YOLO** is used for apple detection and annotation.

### 3. Tomato Maturity

This model identifies the maturity level of tomatoes using **MobileNet**.

### 4. Apple Disease Detection

This model identifies diseases affecting apple leaves using **EfficientNet**.

### 5. Tomato Disease Detection

This model identifies diseases affecting tomato leaves using **MobileNet**.

## Models

| Task                           | Model        |
| ------------------------------ | ------------ |
| Apple & Tomato Detection       | YOLO         |
| Apple & Tomato Classification  | MobileNet    |
| Apple Maturity Classification  | EfficientNet |
| Tomato Maturity Classification | MobileNet    |
| Apple Disease Classification   | EfficientNet |
| Tomato Disease Classification  | MobileNet    |

## Datasets

### Apple & Tomato Detection and Classification

Dataset information will be added later.

### Apple Maturity

The [AppleGrowthVision](https://datacloud.hhi.fraunhofer.de/s/KLFXDw9cWSzXk95?dir=/brandenburg) dataset was used for apple maturity classification.

Three folders were selected based on their acquisition dates:

* `2022-06-27` → Unripe
* `2022-08-03` → Semi-ripe
* `2022-09-06` → Ripe

Apple detection annotations were prepared using Roboflow.

[Apple Detection Annotations – Roboflow](https://app.roboflow.com/hager-hamoda/apple-detection-with-annotation/browse)

### Tomato Maturity

[Tomato Ripness Dataset – Roboflow Universe](https://universe.roboflow.com/postwork/tomato-ripness/browse)

### Tomato Diseases

[Tomato Balanced Dataset – Kaggle]([https://www.kaggle.com/datasets/ghadagsme/tomato-balanced-dataset](https://www.kaggle.com/datasets/ghadagsme/final-tomato-dataset))

### Apple Diseases

[Plant Pathology 2021 – FGVC8 – Kaggle](https://www.kaggle.com/c/plant-pathology-2021-fgvc8)

## Technologies

* Python
* YOLO
* MobileNet
* EfficientNet
* Streamlit
* Roboflow
* Kaggle

## Deployment

The trained models are integrated into a Streamlit application that provides an interface for running the computer vision models and displaying their predictions.

## Team

| Member        | Responsibility                              |
| ------------- | ------------------------------------------- |
| Shahd Abdelhy | Apple & Tomato Detection and Classification |
| Mariam Bahi   | Tomato Maturity                             |
| Hager Ahmed   | Apple Maturity                              |
| Ghada Saeed   | Tomato Disease Detection                    |
| Muhammed Diab | Apple Disease Detection                     |

## Future Improvements

* Improve model performance with additional training data.
* Add support for more plant and disease classes.
* Improve the Streamlit interface.
