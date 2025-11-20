import pandas as pd
import json
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow


# --------------------- VARIABLES DE ENTORNO PARA MLFLOW EN WINDOWS ---------------------
# set MLFLOW_TRACKING_URI=https://dagshub.com/FedeGauna16/labmindat2025.mlflow
# set MLFLOW_TRACKING_USERNAME=<USERNAME>  &  set MLFLOW_TRACKING_PASSWORD=<TOKEN>


def main():
   
   # Cargar parámetros desde el archivo YAML

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]
    model_params = params["train"]["model"]

    # Cargar datos limpios
    df = pd.read_csv("outputs/telco_churn_clean.csv")

    # Dividir datos en características y etiqueta
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Dividir datos en conjuntos de train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Iniciar MLflow
    with mlflow.start_run(run_name=f"logreg_C={model_params.get('C')}"):

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        for p, v in model_params.items():
            mlflow.log_param(p, v)

    # Entrenar el modelo
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    # Registrar métricas en MLflow
    for m, v in metrics.items():
            mlflow.log_metric(m, v)

    # Guardar el modelo
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    mlflow.sklearn.log_model(model, "model")

    # Guardar métricas en un archivo JSON
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()