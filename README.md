# Fraud Review Detection API

This project demonstrates an end-to-end machine learning workflow for detecting potentially fraudulent online reviews using natural language processing.

Note: This repository includes a small sample dataset for reproducibility and quick testing. The full training workflow was also tested locally on a larger dataset containing 2M+ Amazon fake reviews.


## Project Goal

Build a system that supports:
- preprocessing text data
- training classification models
- evaluating model performance
- serving predictions through an API
- containerized deployment

## Project Status

Work in progress

## Planned Technology Stack

- Python
- scikit-learn / PyTorch
- FastAPI
- Docker

## Planned Features

- text preprocessing pipeline
- model training and evaluation
- REST API for predictions
- Docker containerization
- experiment tracking

## Project Structure

```text
fraud-review-detection-api/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── sample/
│   └── processed/
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── api/
│   └── main.py
├── models/
├── tests/
└── artifacts/
```

## Train the Model

- python src/train.py

- The trained model will be saved to:

models/model.pkl
models/vectorizer.pkl

## Run the Project

- Train model: python src/train.py

- Evaluate model: python src/evaluate.py

- Predict example: python src/predict.py

## Run the API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Try:

```bash
Example: http://127.0.0.1:8000/docs
