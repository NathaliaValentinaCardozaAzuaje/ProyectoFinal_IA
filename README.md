# ProyectoFinal_IA — Detección de Fraude con Deep Learning

Sistema de detección de fraude financiero basado en el dataset **Credit Card Fraud Detection** de Kaggle.

## Dataset

El dataset **no está incluido** en este repositorio por su tamaño (143 MB).  
Descárgalo directamente desde Kaggle antes de ejecutar cualquier notebook o la app:

> **[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

Una vez descargado, coloca el archivo `creditcard.csv` en la carpeta `data/`:

```
ProyectoFinal_IA/
└── data/
    └── creditcard.csv   ← aquí
```

---

## Estructura del proyecto

```
ProyectoFinal_IA/
├── app/
│   ├── app.py           # Interfaz Streamlit
│   └── chatbot.py       # Bot conversacional
├── data/
│   └── creditcard.csv   # Dataset (descargar desde Kaggle)
├── models/
│   ├── dnn_regression.pth      # DNN entrenada (PyTorch)
│   ├── regression_model.pkl    # Gradient Boosting (fallback)
│   └── scaler.pkl              # RobustScaler ajustado en train
├── notebooks/
│   └── regression.ipynb  # Notebook principal
├── reports/                    # Gráficas generadas
├── scripts/
│   └── upgrade_dnn_architecture.py  # Script de actualización de arquitectura DNN
├── regression_plan_trabajo.md
└── descripcion_proyecto_final.md
```

---

## Instalación

```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn streamlit joblib
```

## Ejecutar la app

```bash
streamlit run app/app.py
```

## Ejecutar el notebook

Abre `notebooks/regression.ipynb` en Jupyter y ejecuta **Kernel → Restart & Run All**.

---

## Modelo principal — DNN (FraudDNN v3)

Arquitectura profundizada para superar a los modelos clásicos:

```
Input(11) → Dense(512) → BN → ReLU → Dropout(0.35)
          → Dense(256) → BN → ReLU → Dropout(0.30)
          → Dense(128) → BN → ReLU → Dropout(0.25)
          → Dense(64)  → BN → ReLU → Dropout(0.15)
          → Dense(32)  → BN → ReLU → Dropout(0.10)
          → Dense(16)  → BN → ReLU → Dropout(0.05)
          → Dense(1)   → Sigmoid → risk_score ∈ [0,1]
```

| Componente | Valor |
|---|---|
| Optimizer | AdamW (lr=5e-4, weight_decay=5e-5) |
| Loss | WeightedMSE (fraud_weight=15) |
| Scheduler | ReduceLROnPlateau (patience=8) |
| Early Stopping | patience=20, delta=1e-7 |
| Inicialización | He (kaiming_normal_) |

---

## Chatbot con API (opcional y restringido al proyecto)

El chatbot ahora puede funcionar en modo hibrido:
- `Local (reglas)` si no hay API key.
- `API gemini + contexto del proyecto` si defines `GEMINI_API_KEY`.

Variables de entorno:

```bash
GEMINI_API_KEY=tu_api_key
CHATBOT_USE_API=auto        # auto | true | false
CHATBOT_PROVIDER=gemini
CHATBOT_MODEL=gemini-1.5-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
CHATBOT_TIMEOUT_SEC=20
```

El prompt del bot obliga a responder solo con contexto interno del proyecto y, si no encuentra informacion, indica que no tiene ese dato.
La app carga automaticamente estas variables desde `.env` al iniciar.
