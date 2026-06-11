# House Price Prediction Model using Satellite Imagery

### Finance & AI Research Project

Implementation of a hybrid deep learning framework that combines structured housing data with satellite imagery to predict residential property prices.

---

# Project Overview

This project investigates whether satellite imagery can improve house price prediction when combined with traditional tabular housing features.

The core idea is to capture both:

- Property-level information (size, rooms, location features, etc.)
- Visual neighborhood characteristics extracted from satellite imagery

A hybrid deep learning model is developed as the primary contribution of this project. For comparison, tabular-only and CNN-only baselines are also implemented and evaluated.

---

# Problem Statement

Given housing data and geographic coordinates:

- Predict house prices accurately
- Extract visual information from satellite imagery
- Combine structured and image-based features
- Compare hybrid learning against traditional approaches

The objective is to determine whether neighborhood visual context contributes meaningful information beyond tabular features alone.

---

# Model Architecture

The project implements three different approaches:

| Model | Description |
|---------|------------|
| Tabular Model | Uses structured housing features only |
| CNN Model | Uses satellite imagery only |
| Hybrid Model | Combines tabular and image features |

The Hybrid Model serves as the primary model and is used for final predictions.

---

# Methodology

1. Load raw housing datasets
2. Download satellite imagery using property coordinates
3. Clean and preprocess tabular data
4. Perform feature engineering
5. Normalize and scale features
6. Train CNN-only model
7. Train Tabular-only model
8. Train Hybrid Deep Learning model
9. Evaluate model performance
10. Generate predictions and visual explanations

---

# Technologies Used

- Python
- TensorFlow 2.13
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Jupyter Notebook

---

# Repository Structure

```text
House-Price-Prediction-Model/
│
├── datasets/
│   ├── train.csv
│   ├── test.csv
│   ├── train(1).xlsx
│   └── test2.xlsx
│
├── data_fetcher.py
│
├── preprocessing.ipynb
│
├── model_training.ipynb
│
├── final_prediction.csv
│
├── README.md
│
└── requirements.txt
```

---

# File Descriptions

## data_fetcher.py

Downloads satellite images corresponding to each property using latitude and longitude coordinates.

Features:

- Automated image collection
- Configurable image size
- Custom train/test datasets
- Adjustable request delay
- Supports quick testing mode

---

## preprocessing.ipynb

Prepares the dataset for training.

Includes:

- Data cleaning
- Missing value handling
- Feature engineering
- Dataset preparation
- Feature scaling
- Image-path integration

---

## model_training.ipynb

Contains complete model development and evaluation.

Includes:

### Tabular Model

Baseline model trained on structured housing data.

### CNN Model

Image-based model trained solely on satellite imagery.

### Hybrid Model

Primary model that combines:

- CNN image embeddings
- Tabular housing features

Additional functionality:

- Model evaluation
- Performance comparison
- Grad-CAM visualizations
- Explainability analysis

---

# Environment Setup

## Required Environment

This project was developed using:

| Package | Version |
|----------|---------|
| Python | 3.10 |
| TensorFlow | 2.13 |
| NumPy | 1.23.5 |

⚠️ TensorFlow versions 2.14+ may introduce compatibility issues.

---

# Installation

### Create Virtual Environment

```bash
py -3.10 -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# How to Run

## Step 1: Download Satellite Images

```bash
py -3.10 data_fetcher.py
```

Quick test run:

```bash
py -3.10 data_fetcher.py --max-images 10
```

Custom configuration:

```bash
py -3.10 data_fetcher.py \
--train-file datasets/train(1).xlsx \
--test-file datasets/test2.xlsx \
--output-dir satellite_images \
--image-size 256 \
--meters-per-pixel 0.5 \
--delay 0.1
```

---

## Step 2: Preprocess Dataset

Run:

```text
preprocessing.ipynb
```

This notebook generates the final processed dataset used for training.

---

## Step 3: Train Models

Run:

```text
model_training.ipynb
```

The notebook includes:

- Tabular-only model
- CNN-only model
- Hybrid model

---

# Results

The Hybrid Model demonstrates the benefit of incorporating satellite imagery into house price prediction.

### Key Findings

- Visual neighborhood information contributes useful predictive signals.
- Hybrid learning outperforms single-modality approaches.
- Satellite imagery provides contextual information unavailable in tabular data.
- Grad-CAM visualizations improve model interpretability.

---

# Notes for Evaluators

- Pretrained models can be loaded directly.
- Retraining is optional.
- CPU execution is supported.
- GPU acceleration is not required.
- Hybrid model implementation is the primary contribution of the project.

---

# Future Improvements

- Vision Transformers (ViT)
- Multi-scale satellite imagery
- Street-view image integration
- Advanced ensemble techniques
- Web deployment for real-time prediction

---

# Author

Aryan Soni 🎓 B.Tech Student, IIT Roorkee

GitHub: https://github.com/Aryan-soni6387
