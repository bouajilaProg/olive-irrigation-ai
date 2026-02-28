# Olive Health AI: Precision Irrigation System

A machine learning-powered system for predicting irrigation needs and stress levels in olive groves using satellite NDVI data and local temperature inputs.

## Overview

This project provides a model-agnostic web service and API that translates satellite imagery from Google Earth Engine (Sentintel-2) into actionable irrigation recommendations. It supports interchangeable machine learning backends including Random Forest, SVM, LightGBM, and XGBoost.

## Key Features

- **Satellite Integration:** Automated NDVI extraction from Sentinel-2 imagery via Google Earth Engine.
- **Multi-Model Support:** Trainers for Random Forest, Support Vector Machines (SVM), LightGBM, and XGBoost.
- **Model Agnostic Inference:** The core prediction engine loads whichever model is currently marked as "best" or default in the data directory.
- **Automated Reporting:** Each training run generates a JSON report containing model parameters, accuracy scores, and feature importances.
- **Web UI:** Interactive Leaflet-based map interface for selecting grove coordinates and viewing health metrics.
- **Training API:** REST endpoints to trigger model retraining and hyperparameter optimization.

## Project Structure

- `src/app.py`: FastAPI application serving the UI and API.
- `src/olive_brain.py`: Core prediction logic and NDVI-to-feature mapping.
- `src/random_forest/`, `src/svm/`, `src/lgbm/`, `src/xgb/`: Specialized trainers for each algorithm.
- `data/`: Contains the primary dataset (Growth.xlsx) and active model artifacts.
- `runs/`: Historical training run logs and a backup of the best-performing models.

## Installation

1. Clone the repository.

2. Create and activate a environment:

### Using Conda
```bash
conda create -n olive-env python=3.10
conda activate olive-env
pip install -r requirements.txt
```

### Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Authenticate Google Earth Engine:
```bash
python src/auth.py
```

## Usage

### Running the Application
Start the FastAPI server:
```bash
python app.py
```
Access the UI at `http://localhost:8000`.

### Training Models
Retrain models via API endpoints:
- `GET /train/rf`: Random Forest
- `GET /train/svm`: SVM
- `GET /train/lgbm`: LightGBM
- `GET /train/xgb`: XGBoost

### API Prediction
```bash
GET /api/predict?lat=[LATITUDE]&lon=[LONGITUDE]
```

## Data Requirements
The system expects `data/Growth.xlsx` with columns for Temperature, Average_SPAD, and Canopy Cover to train the irrigation classifiers.
