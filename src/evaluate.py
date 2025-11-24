import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import os
import json

def main():
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv("outputs/telco_churn_clean.csv")
    X = df.drop(columns=["churn"])
    y = df["churn"]

    model = joblib.load("models/model.joblib")

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    roc_auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    metrics = {
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

    with open("reports/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("reports/roc_curve.png")
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("reports/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    main()
