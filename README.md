# ADHD Prediction through Eye-Tracking and Speech Analysis

This repository contains code and resources for predicting Attention Deficit Hyperactivity Disorder (ADHD) using a multimodal machine learning approach that integrates eye-tracking metrics and speech features.

## Project Overview

ADHD is a common neurodevelopmental disorder characterized by inattention, hyperactivity, and impulsivity. Early and accurate diagnosis is crucial for effective intervention. This project aims to leverage behavioral signals — specifically eye movement patterns and speech characteristics — to build predictive models for ADHD detection.

## Features

- **Eye-Tracking Data Collection**: Scripts to capture eye movement metrics such as fixation duration, saccadic amplitude, and velocity.
- **Speech Feature Extraction**: Extraction of speech attributes including speech rate and pitch variability.
- **Synthetic Dataset Generation**: Tools to generate synthetic data for model training and evaluation.
- **Model Training & Evaluation**: Machine learning pipeline to train classifiers that distinguish ADHD from control groups.
- **Explainability & Analysis**: Basic interpretation of model predictions to understand contributing features.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install via pip):

```bash
pip install -r requirements.txt
Usage
Collect or generate eye-tracking and speech data.

Extract relevant features using the provided scripts.

Train the ADHD prediction model:

bash
Copy
Edit
python train_model.py
Evaluate model performance:

bash
Copy
Edit
python evaluate_model.py
Repository Structure
data/ — Contains synthetic and sample datasets.

scripts/ — Data collection and preprocessing scripts.

models/ — Model training and evaluation code.

notebooks/ — Exploratory data analysis and experiments.

requirements.txt — Project dependencies.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
