import joblib


model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def predict(text):

    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]

    return prediction