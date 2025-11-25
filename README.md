📄 Reporte Final — Proyecto MLOps con DVC, Pipelines y Despliegue
1. Introducción

En este proyecto se implementa un flujo de trabajo completo de MLOps para un problema de predicción de churn en telecomunicaciones, desde la adquisición de datos hasta la evaluación final del modelo y su preparación para despliegue.
El objetivo principal fue construir un pipeline reproducible, versionable y escalable utilizando DVC, Git, experimentos y prácticas de ingeniería de datos y machine learning.

El pipeline final incluirá:

* Preparación y limpieza del dataset (stage preprocess)

* Entrenamiento modular del modelo (stage train)

* Evaluación avanzada con métricas y visualizaciones (stage evaluate)

* Versionado de datos, modelos y artefactos con DVC

* Control de experimentos y ramas colaborativas

* Integración con DAGsHub como remote para almacenamiento

* Plan de despliegue mediante FastAPI o Streamlit

## Dataset

- Nombre: telco_churn.csv

- Registros: 10.000 clientes

- Variables: 13 columnas

- Variable Objetivo: churn (0 = cliente activo, 1 = se dio de baja)

## Variables principales
- customer_id: identificador único
- age: edad del cliente
- gender: género (Male, Female)
- region: región (North, South, East, West)
- contract_type: tipo de contrato (Month-to-Month, One year, Two year)
- tenure_months: meses de antigüedad
- monthly_charges: cargo mensual
- total_charges: total pagado
- internet_service: (DSL, Fiber optic, No)
- phone_service: (Yes, No)
- multiple_lines: (Yes, No, No phone service)
- payment_method: (Electronic check, Mailed check, Credit card, Bank transfer)

2. Estructura del Proyecto

El proyecto quedó organizado con las siguientes carpetas principales:

├── data/                  # Datos crudos o fuentes externas
├── outputs/               # Datos limpios generados por preprocess
├── models/                # Modelos entrenados
├── reports/               # Métricas, curvas ROC y artefactos de evaluación
├── src/                   # Código del pipeline (preprocess/train/evaluate)
├── dvc.yaml               # Pipeline completo
├── params.yaml            # Hiperparámetros modificables
└── .dvc/                  # Config DVC (cache local + remote config)


Los stages del pipeline son:

- preprocess: limpieza y transformación del dataset

- train: entrenamiento del modelo parametrizable vía params.yaml

- evaluate: métricas extendidas y visualizaciones

La ejecución completa queda automatizada mediante:

dvc repro

3. Comparación de Experimentos

Se ejecutaron 3 experimentos con distinto valor C y se comparan los hiperparámetros.

| Experimento	| C	| Accuracy	| Recall	| Precision | F1-Score |
| --- | --- | --- | --- | --- | --- |
| nosy-gnat-791 |	0.1 |	69%	| 44.76%	| 57.87% | 50.48% |
| gentle-mink-429 |	1.0	|	69.05%	| 41.78%	| 58.65% | 48.80% |
| salty-stork-645 | 10	| 68.65%	| 47.02%	| 56.75% | 51.43% |

Modelo elegido: "salty-stork-645"

* Mayor valor de Recall (47.02%)
* Mayor Valor de F1-Score (51.43%)
* Menor diferencia con el siguiente mejor en Accuracy y Precision
* En general, el experimento con mejor balance entre los hiperparámetros

4. Justificación del Modelo Final

Se mejoró la calidad del proyecto incorporando ramas feat-* para extender la experimentación. En estas ramas se evaluaron dos mejoras principales:
* la inclusión de un modelo Random Forest
* la aplicación de técnicas de feature engineering

Luego, se comparó el rendimiento del modelo base (Logistic Regression) con:
* el modelo Random Forest
* una versión mejorada de Logistic Regression + Feature Engineering.

Los resultados completos e incluyen métricas, conclusiones y visualizaciones generadas en la etapa de evaluación. Se encuentran documentados en el archivo EXPERIMENTS.md.

En resumen.

| Experimento	| C	| Accuracy	| Recall	| Precision | F1-Score |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 10	| 69.65%	| 43.52%	| 62.06% | 51.16% |

Comparando los modelos se determina que el modelo final va a ser la version mejorada de Logistic Regression.

5. Despliegue en Entorno Productivo

La infomación completa del despliegue en producción se puede ver en DEPLOYMENT.md

En resumen: 
 - Arquitecturas propuestas, FastAPI o Streamlit.
 - Versionado del modelo con DVC.
 - Monitoreo continuo de métricas (MLflow).
 - Automatización con CI/CD.
 - Escalado con contenedores (Docker + Kubernetes).
