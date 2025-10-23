import pandas as pd
import json
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def main():
   
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]
    model_params = params["train"]["model"]

    
    df = pd.read_csv("outputs/telco_churn_clean.csv")

    
    X = df.drop(columns=["churn"])
    y = df["churn"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()