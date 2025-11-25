# Despliegue del modelo de Churn — TelcoVision

Este documento describe cómo se podría integrar el modelo entrenado en un servicio real de predicción utilizando FastAPI o Streamlit.

---

## 1. Cargar modelo en producción

El modelo final está almacenado en:
models/model.joblib
y se carga con:

import joblib
model = joblib.load("models/model.joblib")

---

## 2. API REST con FastAPI (opción 1)

FastAPI es una API que permite que cualquier sistema pueda enviar datos y recibir la predicción:

Ejemplo básico con FastAPI:

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model.joblib")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"churn_prediction": int(pred)}

Ejecutar con:
uvicorn app:app --reload

---

## 3. App interactiva con Streamlit (opción 2)

Streamlit permite usar una demo visual:

Ejemplo simple:

import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/model.joblib")
st.title("Predicción de Churn TelcoVision")

age = st.number_input("Age", min_value=18, max_value=90)
monthly_charges = st.number_input("Monthly Charges")

inputs = {"age": age, "monthly_charges": monthly_charges}
if st.button("Predecir"):
    df = pd.DataFrame([inputs])
    pred = model.predict(df)[0]
    st.write("Churn:", pred)

Ejecutar con:
streamlit run app.py

---

## 4. Consideraciones para despliegue real

 * Versionado del modelo y los datos (DVC)
   
   - Permite saber exactamente qué versión del modelo se usó, qué datos, qué parámetros, etc
  
 * Registro y seguimiento en MLflow
   - MLflow guarda:
     . métricas
     . parámetros
     . artefactos
     . gráficos

    Esto es importante para monitorear si el modelo se degrada con el tiempo.
  
  * Automatización con CI/CD
    
   - Con GitHub Actions cada commit reproduce el pipeline, valida que todo funcione y evita "merges" que rompan el entrenamiento

  * Contenedores y despliegue
   - Para un entorno más grande, suele usarse:
     . Docker
     . Kubernetes
     
     Esto hace que el modelo sea fácil de mover entre servidores sin depender del sistema operativo o instalaciones locales.
