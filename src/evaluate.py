import os
import json
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess


def evaluate_model():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "data", "processed", "test_data.csv")
    model_path = os.path.join(repo_root, "models", "model.pkl")
    vectorizer_path = os.path.join(repo_root, "models", "vectorizer.pkl")
    artifacts_path = os.path.join(repo_root, "artifacts", "metrics.json")

    df = load_and_preprocess(data_path)

    X_test = df["clean_text"]
    y_test = df["label"]

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    print("Evaluation Metrics")
    print("------------------")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print("----------------")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.join(repo_root, "artifacts"), exist_ok=True)
    with open(artifacts_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved metrics to {artifacts_path}")


if __name__ == "__main__":
    evaluate_model()