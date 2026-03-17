import os
import joblib
from preprocess import clean_text


def load_artifacts():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "models", "model.pkl")
    vectorizer_path = os.path.join(repo_root, "models", "vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def predict(text: str):
    model, vectorizer = load_artifacts()

    cleaned_text = clean_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(text_vectorized)[0].max()
    else:
        probability = None
    
    label_map = {
        0: "Non-Fake",
        1: "Fake"
    }
    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "prediction": int(prediction),
        "predicted_label": label_map.get(int(prediction), "unknown"),
        "probability": float(probability) if probability is not None else None
    }


if __name__ == "__main__":
    sample_text = "This product is absolutely amazing and works perfectly!!!"
    result = predict(sample_text)
    print(result)