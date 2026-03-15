from sklearn.metrics import classification_report
import joblib


def evaluate(y_true, y_pred):

    report = classification_report(y_true, y_pred)

    print(report)