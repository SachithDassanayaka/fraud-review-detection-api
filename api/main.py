import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict

# allow importing from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

app = FastAPI(title="Fraud Review Detection API")


class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Fraud Review Detection API is running"}


@app.post("/predict")
def predict_review(request: ReviewRequest):
    result = predict(request.text)
    return result
