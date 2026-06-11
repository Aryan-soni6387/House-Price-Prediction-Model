House Price Prediction Model using Satellite Imagery
Project Overview

This project focuses on predicting house prices using a hybrid deep learning approach that combines structured tabular housing data with satellite imagery.
The key idea is to capture both property-level attributes (such as size, rooms, location features) and visual neighborhood context extracted from satellite images, which often has a significant impact on real estate prices.

The hybrid model is the primary contribution of this project, while tabular-only and CNN-only models are included for comparison and analysis.
```text
House Price Prediction Model/
    │
    ├── datasets/
    |   └── Raw tabular housing datasets
    │
    ├── data_fetcher.py
    |   └── Downloads satellite images using property coordinates
    │
    ├── preprocessing.ipynb
    |   └── Data cleaning, feature engineering, and dataset preparation
    │
    ├── model_training.ipynb
    |   └── Contains tabular, CNN-only, and hybrid models (primary model)
    |
    ├── best_hybrid_model_final.h5
    |   └── Final trained hybrid model (primary model)
    |
    ├── cnn_training.h5
    │   └── Trained CNN-only model
    │
    ├── README.md
    │
    └── requirements.txt


File Descriptions

data_fetcher.py
    This script is responsible for fetching satellite images corresponding to each house using latitude and longitude information.
    Uses satellite imagery services
    Saves images locally in a structured format
    Supports configurable train/test input files, output directory, image size, request delay, and test-size limits through command-line arguments
    Intended to be run before preprocessing and training
    Note: The satellite images directory is not included in the repository due to size constraints.

preprocessing.ipynb
    This notebook prepares the data for modeling.
    It includes:
        Cleaning of raw tabular data
        Feature engineering
        Attaching image paths to each data point
        Scaling and normalization of tabular features
        The output of this notebook is a dataset ready for model training.

model_training.ipynb

    These notebooks contain all modeling logic, including:
        A tabular-only model (For comparision purposes)
        The hybrid model (main model of the project, from cell 13 to cell 25)
        A CNN-only model

    They also include:
        Code for model evaluation
        Grad-CAM / heatmap generation for explainability
        Comparison between different modeling approaches

requirement.txt
    contains all the modules required for the project


⚠️ Important:
    Running these notebooks is optional. Pretrained model files are already provided.
    Directly load the trained models Or retrain the models if desired




Environment & Setup Instructions
Required Environment

This project was developed and tested using:

    Python: 3.10
    TensorFlow: 2.13
    NumPy: 1.23.5
    and other libararies mentioned in the requirements.txt

Important Note
    Newer TensorFlow versions (2.14+) introduce breaking changes and are not compatible with this codebase.


Setup Steps

1).Create a virtual environment using Python 3.10:

    py -3.10 -m venv venv


2).Activate the environment:

    Windows
    venv\Scripts\activate

3).Install all required packages

    pip install -r requirements.txt

4).Running the Project

    Download satellite images

        py -3.10 data_fetcher.py

    Optional quick test run:

        py -3.10 data_fetcher.py --max-images 10

    Optional custom run:

        py -3.10 data_fetcher.py --train-file datasets/train(1).xlsx --test-file datasets/test2.xlsx --output-dir satellite_images --image-size 256 --meters-per-pixel 0.5 --delay 0.1

5).Preprocess the data

    Run preprocessing.ipynb

6).Model training

    Run model_training.ipynb for the hybrid model

    Alternatively, pretrained models can be loaded directly without retraining.


Notes for Evaluators

    Training deep models is optional

    Pretrained checkpoints are provided for direct evaluation

    The project is CPU-compatible; no GPU is required

    The hybrid model is the primary contribution
