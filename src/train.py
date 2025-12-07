import pandas as pd
import json
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow


# --------------------- VARIABLES DE ENTORNO PARA MLFLOW EN WINDOWS ---------------------
# set MLFLOW_TRACKING_URI=https://dagshub.com/FedeGauna16/labmindat2025.mlflow
# set MLFLOW_TRACKING_USERNAME=<USERNAME>  &  set MLFLOW_TRACKING_PASSWORD=<TOKEN>

def train_logistic(params):

    input_path = "outputs/telco_churn_clean.csv"

    df = pd.read_csv(input_path)

    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]
    model_params = params["train"]["logistic_regression"]

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(**model_params)
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    for p, v in model_params.items():
            mlflow.log_param(p, v)
    return model, metrics

def train_random_forest(params):

    input_path = "outputs/telco_churn_clean.csv"

    df = pd.read_csv(input_path)

    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]
    model_params = params["train"]["random_forest"]

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    for p, v in model_params.items():
            mlflow.log_param(p, v)
    return model, metrics

def main():
    
    metrics = {}
    model = None
    # Configurar experimento de MLflow
    mlflow.set_experiment("Predicción_Churn en Telco")

   # Cargar parámetros desde el archivo YAML

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    model_name = params["train"]["model_name"]

    # Iniciar MLflow
    with mlflow.start_run(run_name=model_name):

        if model_name == "logistic_regression":
            model, metrics = train_logistic(params)

        elif model_name == "random_forest":
            model, metrics = train_random_forest(params)

        #mlflow.log_param("test_size", test_size)
        #mlflow.log_param("random_state", random_state)

        #for p, v in model_params.items():
        #    mlflow.log_param(p, v)
        
        # Registrar métricas en MLflow
        for m, v in metrics.items():
                mlflow.log_metric(m, v)

        # Guardar métricas en un archivo JSON
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Guardar el modelo
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")
        mlflow.log_artifact("models/model.joblib")
        mlflow.log_artifact("metrics.json")


if __name__ == "__main__":
    main()