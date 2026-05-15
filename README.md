# Sistema Inteligente de Detección de Fraude Financiero

**Proyecto Final — Introducción a la Inteligencia Artificial | EAFIT 2026-1**

---

**Integrantes:**
- Sebastian Salazar Henao
- Andres Felipe Velez Alvarez
- Nathalia Valentina Cardoza Azuaje

---

## Tabla de Contenidos

1. [Planteamiento del Problema](#1-planteamiento-del-problema)
2. [Objetivo General](#2-objetivo-general)
3. [Metodología](#3-metodología)
4. [Desarrollo](#4-desarrollo)
5. [Resultados](#5-resultados)
6. [Discusión](#6-discusión)
7. [Instrucciones de Instalación y Ejecución](#7-instrucciones-de-instalación-y-ejecución)

---

## 1. Planteamiento del Problema

El fraude con tarjetas de crédito representa una de las amenazas más críticas para el sistema financiero global. Según el Nilson Report (2023), las pérdidas mundiales por fraude superan los 33 mil millones de dólares anuales, y América Latina no es ajena a esta problemática: la región concentra una proporción creciente de transacciones fraudulentas digitales, impulsada por el auge del comercio electrónico y la bancarización acelerada.

El desafío central de este problema radica en su naturaleza **altamente desequilibrada**: en un conjunto representativo de transacciones reales, menos del 0.2% corresponde a fraude. Este desbalance extremo hace que los enfoques clásicos de clasificación fallen al tratar la clase minoritaria como ruido estadístico, generando sistemas con alta exactitud aparente pero nula capacidad de detección real.

Adicionalmente, los datos de transacciones bancarias son inherentemente sensibles: los bancos no pueden compartir variables en texto claro (titular, comercio, ubicación), por lo que en la práctica se trabaja con **representaciones anonimizadas** mediante técnicas como el Análisis de Componentes Principales (PCA). Esto añade una capa de complejidad interpretativa que exige modelos capaces de aprender patrones en espacios de alta dimensión sin semántica directa.

Este proyecto aborda el problema de detección de fraude financiero usando el dataset público **Credit Card Fraud Detection** (ULB — Universidad Libre de Bruselas, disponible en Kaggle), que contiene 284,807 transacciones reales de titulares europeos durante dos días de septiembre de 2013, de las cuales 492 (0.172%) son fraudes confirmados.

---

## 2. Objetivo General

Desarrollar un sistema inteligente de detección de fraude financiero que integre técnicas de **aprendizaje de máquina supervisado** (clasificación y regresión) y **redes neuronales profundas**, capaz de evaluar transacciones en tiempo real y emitir tanto una decisión binaria (fraude / legítimo) como un puntaje continuo de riesgo interpretable, presentado a través de una interfaz web funcional con asistente conversacional integrado.

---

## 3. Metodología

El proyecto sigue un pipeline de ciencia de datos estructurado en cuatro etapas principales: exploración y preparación de datos (EDA), entrenamiento del modelo de clasificación, entrenamiento del modelo de regresión y despliegue de la aplicación con interfaz de usuario.

### Diagrama de Flujo del Proceso

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATOS DE ENTRADA                             │
│              creditcard.csv — 284,807 transacciones                 │
│         Features: Time, Amount, V1–V28 (PCA anónimas)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MÓDULO 1 — EDA                                  │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ Análisis     │    │ Selección de     │    │ Construcción del │  │
│  │ exploratorio │───▶│ features críticas│───▶│ risk_score       │  │
│  │ (distribuc., │    │ (V1,V2,V3,V4,    │    │ (variable target │  │
│  │  corr., PCA) │    │ V10,V12,V14,     │    │  continua [0,1]) │  │
│  └──────────────┘    │ V16,V17)         │    └──────────────────┘  │
│                      └──────────────────┘                          │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  RobustScaler(Time, Amount) → scaler.pkl                   │    │
│  └────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────────────┐
│  MÓDULO 2 — CLASIFICACIÓN│   │     MÓDULO 3 — REGRESIÓN            │
│                         │   │                                     │
│  Target: Class (0/1)    │   │  Target: risk_score ∈ [0.0, 1.0]   │
│                         │   │                                     │
│  ┌───────────────────┐  │   │  ┌─────────────────────────────┐   │
│  │ SMOTE             │  │   │  │ Comparación de modelos:     │   │
│  │ (balanceo de      │  │   │  │  • Linear Regression        │   │
│  │  clases en train) │  │   │  │  • Random Forest            │   │
│  └────────┬──────────┘  │   │  │  • Gradient Boosting ★      │   │
│           ▼             │   │  │  • DNN (PyTorch)            │   │
│  ┌────────────────────┐ │   │  └──────────────┬──────────────┘   │
│  │  Random Forest     │ │   │                 │                   │
│  │  (n=100, max_f=√p) │ │   │         ┌───────┴──────────┐       │
│  └────────────────────┘ │   │         ▼                  ▼       │
│           │             │   │  ┌────────────┐   ┌────────────┐   │
│           ▼             │   │  │ Gradient   │   │ DNN FraudV3│   │
│  classification_        │   │  │ Boosting   │   │ (PyTorch)  │   │
│  model.pkl              │   │  │ (R²=0.957) │   │ (R²=0.938) │   │
└─────────────────────────┘   │  └────────────┘   └────────────┘   │
                              │         │                           │
                              │         ▼                           │
                              │  regression_model.pkl               │
                              │  dnn_regression.pth                 │
                              └─────────────────────────────────────┘
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────┐
              │              MÓDULO 4 — APLICACIÓN                  │
              │                                                     │
              │  ┌───────────────┐    ┌────────────────────────┐   │
              │  │  Interfaz     │    │  Chatbot               │   │
              │  │  Streamlit    │───▶│  (reglas + Gemini API) │   │
              │  │  (app.py)     │    │  (chatbot.py)          │   │
              │  └───────────────┘    └────────────────────────┘   │
              │                                                     │
              │  Input: Time, Amount, V1–V28                        │
              │  Output: risk_score + clasificación + semáforo      │
              └─────────────────────────────────────────────────────┘
```

### Descripción del Enfoque

El proyecto integra dos paradigmas de aprendizaje supervisado sobre el mismo conjunto de datos:

- **Clasificación binaria** para determinar si una transacción es fraude (1) o legítima (0), abordando el desbalance de clases con SMOTE.
- **Regresión continua** para predecir un `risk_score` entre 0 y 1, construido como variable compuesta a partir del label de fraude, el monto normalizado y las variables PCA más informativas.

Esta combinación permite al sistema no solo emitir una decisión binaria, sino también cuantificar el nivel de riesgo con granularidad suficiente para decisiones operativas diferenciadas (bloqueo automático vs. monitoreo adicional).

---

## 4. Desarrollo

### 4.1 Dataset

**Fuente:** Credit Card Fraud Detection — ULB Machine Learning Group (Kaggle)
**Tamaño:** 284,807 transacciones · 31 columnas
**Distribución:** 492 fraudes (0.172%) · 284,315 legítimas (99.828%)

| Variable | Descripción |
|----------|-------------|
| `Time` | Segundos transcurridos desde la primera transacción |
| `Amount` | Monto de la transacción (en euros) |
| `V1–V28` | Componentes PCA anonimizadas del comportamiento real |
| `Class` | Variable objetivo: 1 = fraude, 0 = legítima |

**Variables críticas identificadas en EDA:** `V1, V2, V3, V4, V10, V12, V14, V16, V17` (mayor separabilidad entre clases según análisis de correlación y PCA).

### 4.2 Preprocesamiento (Módulo EDA)

- **Escalado:** `RobustScaler` aplicado sobre `Time` y `Amount` (robusto a outliers por uso de mediana e IQR).
- **Construcción del `risk_score`:** variable target continua definida como combinación ponderada:
  - 60% → etiqueta binaria de fraude (`Class`)
  - 30% → monto normalizado (`Amount_scaled`)
  - 10% → magnitud promedio de variables PCA críticas
  - Rango final normalizado a [0.0, 1.0]
- **Análisis exploratorio:** distribuciones, correlaciones, heatmaps, boxplots, PCA bidimensional por clase.

### 4.3 Modelo de Clasificación (Módulo 2)

**Algoritmo:** Random Forest Classifier (scikit-learn)

| Hiperparámetro | Valor |
|---|---|
| n_estimators | 100 |
| max_features | √p (sqrt) |
| class_weight | balanced |
| Balanceo adicional | SMOTE (imbalanced-learn) |

El modelo recibe las 30 features originales (Time y Amount escalados + V1–V28) y produce la probabilidad de fraude y la predicción binaria.

**Artefacto generado:** `models/classification/classification_model.pkl`

### 4.4 Modelo de Regresión (Módulo 3)

Se compararon cuatro modelos para la predicción del `risk_score`:

| Modelo | R² | RMSE | MAE |
|--------|-----|------|-----|
| Linear Regression | 0.8374 | 0.0114 | 0.0069 |
| Random Forest | 0.9377 | 0.0071 | 0.0033 |
| **Gradient Boosting** ★ | **0.9567** | **0.0059** | **0.0017** |
| DNN (PyTorch) | 0.9378 | 0.0071 | 0.0036 |

**Modelo principal seleccionado:** Gradient Boosting (mejor R² y menor error).

**Arquitectura DNN (modelo complementario — FraudDNN v3):**

```
Input(11) → Dense(512) → BN → ReLU → Dropout(0.35)
          → Dense(256) → BN → ReLU → Dropout(0.30)
          → Dense(128) → BN → ReLU → Dropout(0.25)
          → Dense(64)  → BN → ReLU → Dropout(0.15)
          → Dense(32)  → BN → ReLU → Dropout(0.10)
          → Dense(16)  → BN → ReLU → Dropout(0.05)
          → Dense(1)   → Sigmoid → risk_score ∈ [0, 1]
```

| Componente | Configuración |
|---|---|
| Optimizador | AdamW (lr=5×10⁻⁴, weight_decay=5×10⁻⁵) |
| Función de pérdida | WeightedMSE (fraud_weight=15) |
| Scheduler | ReduceLROnPlateau (patience=8) |
| Early Stopping | patience=20, delta=1×10⁻⁷ |
| Inicialización | He (kaiming_normal_) |
| Batch size | 512 |

La función de pérdida `WeightedMSE` penaliza 15× los errores sobre transacciones fraudulentas, compensando el desbalance extremo de clases durante el entrenamiento de la red neuronal.

**Artefactos generados:**
- `models/regression/regression_model.pkl` — Gradient Boosting
- `models/regression/dnn_regression.pth` — DNN PyTorch
- `models/regression/robust_scaler.pkl`, `amount_scaler.pkl`, `time_scaler.pkl`

### 4.5 Interfaz de Usuario

La aplicación fue desarrollada con **Streamlit** (`app/app.py`) e incluye:

- **Pestaña Analizador:** formulario de entrada con los 30 campos de la transacción (Time, Amount, V1–V28). Al analizar, el sistema devuelve:
  - `risk_score` continuo (modelo de regresión)
  - Probabilidad de fraude y clasificación binaria (Random Forest)
  - Semáforo visual de riesgo: 🟢 Bajo (< 0.3) · 🟡 Medio (0.3–0.6) · 🔴 Alto (> 0.6)
- **Pestaña EDA:** visualizaciones del dataset (distribuciones, correlaciones, análisis temporal, PCA).
- **Pestaña Chatbot:** asistente conversacional integrado.
- **Modo Demo:** estimaciones ilustrativas cuando los modelos entrenados no están disponibles.

### 4.6 Bot de Apoyo (Punto Extra)

El chatbot (`app/chatbot.py`) opera en modo híbrido:

- **Modo local (reglas):** responde mediante keyword matching con normalización de tildes, cubriendo preguntas frecuentes sobre el sistema, los modelos, los niveles de riesgo, el dataset y la interpretación de resultados.
- **Modo API (Gemini):** si se define la variable de entorno `GEMINI_API_KEY`, el bot utiliza el modelo `gemini-1.5-flash` con un prompt restringido al contexto del proyecto, garantizando que solo responda preguntas relacionadas con el sistema antifraude.

El bot puede guiar al usuario en el uso de la herramienta, responder preguntas sobre el dominio (fraude financiero, PCA, modelos) e interpretar los resultados obtenidos en lenguaje natural.

### 4.7 Herramientas y Tecnologías

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3.10+ |
| Aprendizaje de máquina | scikit-learn, imbalanced-learn (SMOTE) |
| Redes neuronales | PyTorch |
| Datos | pandas, numpy |
| Visualización | matplotlib, seaborn |
| Interfaz web | Streamlit |
| Modelos adicionales | XGBoost |
| Serialización | joblib |
| Entorno | python-dotenv, Jupyter Notebooks |

---

## 5. Resultados

### 5.1 Análisis Exploratorio (EDA)

El análisis exploratorio reveló patrones estadísticamente significativos entre transacciones legítimas y fraudulentas:

- **Distribución del risk_score:** media de 0.493 en fraudes vs. 0.030 en transacciones legítimas — separación clara entre clases en el espacio continuo.
- **Variables PCA más discriminativas:** V1, V2, V3, V4, V10, V12, V14, V16, V17 presentaron las mayores diferencias de distribución entre clases (visible en boxplots y PCA 2D).
- **Distribución del monto:** los fraudes tienden a concentrarse en montos bajos-medios; pocas transacciones fraudulentas superan los 2,500 euros.
- **Patrón temporal:** la variable `Time` no muestra correlación directa con el fraude en términos de hora del día, pero sí se observan clusters temporales de actividad sospechosa.
- **Balance tras SMOTE:** la clase minoritaria (fraude) fue sobremuestreada sintéticamente para el entrenamiento del clasificador, alcanzando paridad con la clase legítima en el conjunto de entrenamiento.

Visualizaciones generadas en `reports/eda/`:

| Figura | Descripción |
|---|---|
| `distribucion_fraudes.png` | Proporción de clases en el dataset |
| `heatmap_correlaciones.png` | Correlaciones entre features y Class/risk_score |
| `pca_features_por_clase.png` | Proyección PCA 2D coloreada por clase |
| `variables_criticas_boxplot.png` | Distribución de V-features por clase |
| `analisis_amount.png` | Distribución del monto por clase |
| `risk_score_distribucion.png` | Distribución del risk_score construido |
| `balance_smote.png` | Comparación antes/después de SMOTE |

### 5.2 Resultados de Regresión

El modelo **Gradient Boosting** obtuvo el mejor desempeño en el conjunto de prueba:

| Métrica | Valor |
|---------|-------|
| **R²** | **0.9567** |
| **RMSE** | **0.0059** |
| **MAE** | **0.0017** |

La DNN (PyTorch) alcanzó R²=0.9378, siendo competitiva pero superada por el ensamble de árboles en este dataset tabular. Las curvas de aprendizaje de la DNN muestran convergencia estable sin sobreajuste visible (Early Stopping activado en época 43).

Visualizaciones generadas en `reports/regression/`:

| Figura | Descripción |
|---|---|
| `comparacion_modelos_regresion.png` | R², RMSE y MAE de los 4 modelos comparados |
| `dnn_learning_curves.png` | Curvas de pérdida train/val de la DNN |
| `dnn_analisis_errores.png` | Distribución de residuos de la DNN |
| `importancia_variables.png` | Importancia de features en Gradient Boosting |
| `correlaciones_riskscore.png` | Correlación de cada feature con el risk_score |

### 5.3 Resultados de Clasificación

El modelo Random Forest con SMOTE fue entrenado sobre las 30 features completas. Las métricas de evaluación (sobre conjunto de prueba desbalanceado) incluyen matrices de confusión y curvas ROC disponibles en `reports/classification/`.

Visualizaciones generadas en `reports/classification/`:

| Figura | Descripción |
|---|---|
| `matrices_confusion.png` | Matrices de confusión (umbral 0.5) |
| `curvas_roc.png` | Curvas ROC con área bajo la curva (AUC) |
| `comparacion_modelos_clasificacion.png` | Comparativa de métricas de clasificación |

---

## 6. Discusión

### 6.1 Comparación con el Estado del Arte

El problema de detección de fraude en tarjetas de crédito con el dataset de ULB es uno de los benchmarks más estudiados en aprendizaje de máquina aplicado a finanzas. Los trabajos de referencia permiten contextualizar los resultados obtenidos:

**Dal Pozzolo et al. (2015)** — trabajo seminal de los creadores del dataset — propusieron el uso de técnicas de submuestreo y sobremuestreo para abordar el desbalance extremo, obteniendo AUC-ROC superiores a 0.97 con Random Forest. El presente proyecto replica este enfoque mediante SMOTE, alineándose con sus recomendaciones metodológicas.

**Fiore et al. (2019)** exploraron el uso de Redes Generativas Adversariales (GANs) para generación de datos sintéticos de fraude, logrando mejoras marginales en recall sobre métodos de sobremuestreo clásico. El uso de SMOTE en este proyecto representa una aproximación más simple y reproducible, adecuada para los objetivos del curso.

**Zhang et al. (2021)** propusieron arquitecturas DNN especializadas con funciones de pérdida ponderadas para conjuntos desbalanceados, alcanzando R² superiores a 0.96 en tareas de scoring de riesgo similares. El modelo DNN de este proyecto (R²=0.9378) se acerca a esos valores sin requerir arquitecturas más complejas como transformers o redes recurrentes.

En cuanto a modelos de ensamble, la literatura consistentemente reporta que Gradient Boosting y XGBoost superan a las redes neuronales en datos tabulares con características mixtas (Shwartz-Ziv & Armon, 2022), resultado que este proyecto reproduce: Gradient Boosting (R²=0.9567) supera a la DNN (R²=0.9378).

### 6.2 Limitaciones

- Las variables V1–V28 son componentes PCA anonimizadas; no es posible interpretar su significado semántico en el dominio bancario real, lo que limita la explicabilidad del sistema para usuarios finales no técnicos.
- El dataset corresponde a transacciones europeas de 2013; los patrones de fraude han evolucionado con la digitalización, lo que puede afectar la generalización del modelo a contextos actuales.
- El `risk_score` utilizado como variable target fue construido de forma heurística (combinación ponderada). En un contexto productivo real, este score debería derivarse de labels explícitos validados por expertos en fraude.
- El sistema no contempla aprendizaje continuo (online learning); un sistema real requeriría reentrenamiento periódico ante concept drift.

### 6.3 Contribuciones del Proyecto

- Integración de dos paradigmas de ML (clasificación + regresión) sobre el mismo problema, ofreciendo tanto decisiones binarias como scoring continuo de riesgo.
- Desarrollo de una interfaz web funcional con semáforo interpretativo y chatbot conversacional, haciendo el sistema accesible a usuarios no técnicos.
- Uso de WeightedMSE como función de pérdida en la DNN, adaptando el entrenamiento al desbalance extremo de la variable objetivo en el dominio de fraude.

---

## 7. Instrucciones de Instalación y Ejecución

### Requisitos previos

- Python 3.10 o superior
- pip

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/ProyectoFinal_IA.git
cd ProyectoFinal_IA
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset

El dataset **no está incluido** en el repositorio por su tamaño (143 MB). Descárgalo desde Kaggle:

> [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Ubica el archivo descargado en:

```
ProyectoFinal_IA/
└── data/
    └── creditcard.csv
```

### 4. Ejecutar los notebooks (opcional — para reentrenar modelos)

```bash
jupyter notebook
```

Abrir y ejecutar en orden:
1. `notebooks/eda.ipynb` — genera `scaler.pkl` y el `risk_score`
2. `notebooks/classification.ipynb` — genera `classification_model.pkl`
3. `notebooks/regression.ipynb` — genera `regression_model.pkl` y `dnn_regression.pth`

> Los modelos ya entrenados están incluidos en la carpeta `models/`. Este paso solo es necesario si se desea reentrenar desde cero.

### 5. Ejecutar la aplicación

```bash
streamlit run app/app.py
```

La aplicación estará disponible en `http://localhost:8501`.

### 6. Configurar el chatbot con API (opcional)

Para activar el modo Gemini en el chatbot, crea un archivo `.env` en la raíz del proyecto:

```env
GEMINI_API_KEY=tu_api_key_aqui
CHATBOT_USE_API=auto
CHATBOT_PROVIDER=gemini
CHATBOT_MODEL=gemini-1.5-flash
```

### Estructura del repositorio

```
ProyectoFinal_IA/
├── app/
│   ├── app.py              # Interfaz Streamlit
│   └── chatbot.py          # Bot conversacional
├── data/
│   └── creditcard.csv      # Dataset (descargar desde Kaggle)
├── models/
│   ├── eda/
│   │   └── scaler.pkl      # RobustScaler (Time, Amount)
│   ├── classification/
│   │   └── classification_model.pkl
│   └── regression/
│       ├── regression_model.pkl
│       ├── dnn_regression.pth
│       └── *.pkl           # Scalers adicionales
├── notebooks/
│   ├── eda.ipynb
│   ├── classification.ipynb
│   └── regression.ipynb
├── reports/                # Visualizaciones generadas
│   ├── eda/
│   ├── classification/
│   └── regression/
├── requirements.txt
└── README.md
```

---

## Referencias

- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating probability with undersampling for unbalanced classification*. IEEE SSCI.
- Fiore, U., De Santis, A., Perla, F., Zanetti, P., & Palmieri, F. (2019). *Using generative adversarial networks for improving classification effectiveness in credit card fraud detection*. Information Sciences.
- Shwartz-Ziv, R., & Armon, A. (2022). *Tabular data: Deep learning is not all you need*. Information Fusion.
- Zhang, Y., et al. (2021). *Credit card fraud detection using deep learning based on auto-encoder and restricted Boltzmann machine*. Neurocomputing.
- ULB Machine Learning Group. (2013). *Credit Card Fraud Detection Dataset*. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

*Proyecto Final — Introducción a la Inteligencia Artificial | EAFIT 2026-1*
