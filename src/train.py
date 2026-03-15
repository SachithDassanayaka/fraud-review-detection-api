import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

from preprocess import load_and_preprocess


def train_model():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "data", "sample", "reviews_sample.csv")
    models_dir = os.path.join(repo_root, "models")

    df = load_and_preprocess(data_path)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))

    print("\nSaved model and vectorizer.")


if __name__ == "__main__":
    train_model()