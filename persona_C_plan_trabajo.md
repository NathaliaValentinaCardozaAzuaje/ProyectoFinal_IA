# Plan de Trabajo — Persona C
## Proyecto: Sistema Inteligente de Detección y Análisis de Fraude Financiero
### Curso: Introducción a la Inteligencia Artificial — EAFIT 2026-1

---

## Contexto del proyecto

Eres parte de un equipo de tres personas desarrollando un sistema de detección de fraude financiero. Tu rol cubre tres componentes: modelo de regresión, interfaz Streamlit y bot conversacional.

**Dataset:** `creditcard.csv` — contiene transacciones bancarias con variables anonimizadas (V1–V28), `Amount`, `Time` y `Class` (0 = legítima, 1 = fraude).

**Tu trabajo es completamente independiente.** No dependes del preprocesamiento ni del notebook de Persona A. Persona B te entregará `classification_model.pkl` al final; mientras tanto usarás modelos simulados.

**Archivos que debes producir:**
- `notebooks/persona_C_regression_app_bot.ipynb`
- `models/regression_model.pkl`
- `app/app.py`
- `app/chatbot.py`

---

## Parte A — Modelo de Regresión

### Objetivo
Construir un modelo que prediga un puntaje de riesgo continuo (`risk_score`) por transacción, como complemento al clasificador binario de Persona B.

### Instrucciones paso a paso

#### Paso 1 — Cargar el dataset

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/creditcard.csv')
print(df.shape)
print(df.head())
```

#### Paso 2 — Crear la variable objetivo `risk_score`

No existe en el dataset original. Debes crearla como una combinación de variables existentes que capture el nivel de riesgo de cada transacción.

**Fórmula sugerida:**

```python
df['risk_score'] = (
    df['Class'] * 0.6 +
    (df['Amount'] / df['Amount'].max()) * 0.3 +
    df[['V1','V2','V3','V4','V10','V12','V14','V16','V17']].abs().mean(axis=1) * 0.1
)
```

> Esta fórmula pondera: si es fraude confirmado (60%), el monto normalizado (30%) y la magnitud promedio de las variables más correlacionadas con fraude (10%). Puedes ajustar los pesos si encuentras una mejor combinación.

#### Paso 3 — Preprocesamiento

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Seleccionar features
features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V10', 'V12', 'V14', 'V16', 'V17']
X = df[features]
y = df['risk_score']

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

#### Paso 4 — Entrenar modelos

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

#### Paso 5 — Evaluar y comparar

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

for nombre, modelo in [("Linear Regression", lr), ("Random Forest", rf)]:
    y_pred = modelo.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"{nombre}: RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")
```

#### Paso 6 — Exportar el mejor modelo

```python
import joblib

# Guardar el modelo con mejor R²
joblib.dump(rf, 'models/regression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Modelos guardados correctamente.")
```

---

## Parte B — Interfaz Streamlit

### Objetivo
Construir `app/app.py`: una aplicación web funcional e intuitiva que permita ingresar datos de una transacción y obtener predicciones de ambos modelos (regresión y clasificación).

### Estructura general de app.py

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Sistema Antifraude", layout="wide")
```

### Instrucciones paso a paso

#### Paso 7 — Cargar modelos

Usa modelos simulados primero. Cuando Persona B entregue su archivo, reemplaza solo las líneas de carga.

```python
# --- FASE 1: modelos dummy (usar mientras B termina) ---
class ModeloDummy:
    def predict(self, X):
        return np.random.uniform(0, 1, len(X))
    def predict_proba(self, X):
        p = np.random.uniform(0, 1, len(X))
        return np.column_stack([1-p, p])

regression_model    = ModeloDummy()
classification_model = ModeloDummy()

# --- FASE 2: reemplazar cuando B entregue el .pkl ---
# regression_model     = joblib.load('models/regression_model.pkl')
# classification_model = joblib.load('models/classification_model.pkl')
# scaler               = joblib.load('models/scaler.pkl')
```

#### Paso 8 — Formulario de entrada

```python
st.title("Sistema de Detección de Fraude Financiero")
st.subheader("Ingresa los datos de la transacción")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Monto de la transacción (USD)", min_value=0.0, value=150.0)
    time   = st.number_input("Tiempo desde primera transacción (segundos)", min_value=0.0, value=50000.0)

with col2:
    v1  = st.slider("V1",  -5.0, 5.0, 0.0)
    v2  = st.slider("V2",  -5.0, 5.0, 0.0)
    v14 = st.slider("V14", -5.0, 5.0, 0.0)
    v17 = st.slider("V17", -5.0, 5.0, 0.0)
```

> Puedes simplificar el formulario mostrando solo las variables más interpretables (Amount, Time) y dejando las V como sliders avanzados en un `st.expander("Parámetros avanzados")`.

#### Paso 9 — Mostrar resultados

```python
if st.button("Analizar transacción"):
    entrada = np.array([[time, amount, v1, v2, 0, 0, 0, 0, 0, 0, v14, 0, v17]])

    risk_score   = regression_model.predict(entrada)[0]
    prob_fraude  = classification_model.predict_proba(entrada)[0][1]

    col_r, col_c = st.columns(2)

    with col_r:
        st.metric("Puntaje de riesgo", f"{risk_score:.2f}")
        st.progress(float(risk_score))

    with col_c:
        st.metric("Probabilidad de fraude", f"{prob_fraude*100:.1f}%")
        if prob_fraude > 0.5:
            st.error("⚠️ ALERTA: Transacción sospechosa de fraude")
        else:
            st.success("✅ Transacción probablemente legítima")
```

---

## Parte C — Bot Conversacional

### Objetivo
Construir `app/chatbot.py` con la lógica del bot, e integrarlo en `app.py` usando los componentes de chat de Streamlit.

### Instrucciones paso a paso

#### Paso 10 — Crear chatbot.py

```python
# app/chatbot.py

RESPUESTAS = {
    "riesgo alto": (
        "Un riesgo alto significa que el modelo detectó patrones similares "
        "a transacciones fraudulentas. No garantiza fraude, pero recomienda "
        "verificación adicional."
    ),
    "riesgo bajo": (
        "Un riesgo bajo indica que la transacción tiene características "
        "similares a operaciones legítimas. El sistema la considera segura."
    ),
    "fraude": (
        "El sistema detecta fraude analizando patrones en el monto, el tiempo "
        "y variables derivadas del comportamiento de la tarjeta. Cuando estos "
        "patrones coinciden con fraudes históricos, se emite una alerta."
    ),
    "risk score": (
        "El risk_score es un valor entre 0 y 1 que representa qué tan riesgosa "
        "es una transacción. Cercano a 0 = bajo riesgo. Cercano a 1 = alto riesgo."
    ),
    "qué hace": (
        "Este sistema analiza transacciones financieras en tiempo real usando "
        "modelos de machine learning para identificar posibles fraudes."
    ),
    "modelo": (
        "Usamos dos modelos: uno de clasificación (fraude/no fraude) y uno de "
        "regresión (puntaje continuo de riesgo). Ambos fueron entrenados con "
        "datos reales anonimizados."
    ),
    "variables": (
        "Las variables V1 a V28 son el resultado de una transformación PCA "
        "aplicada a datos reales para proteger la privacidad de los usuarios. "
        "Amount y Time son las únicas variables sin transformar."
    ),
}

RESPUESTA_DEFAULT = (
    "No tengo una respuesta para eso aún. Puedes preguntarme sobre: "
    "riesgo alto, riesgo bajo, fraude, el risk score, qué hace el sistema, "
    "los modelos usados o las variables del dataset."
)

def responder(pregunta: str) -> str:
    pregunta = pregunta.lower()
    for clave, respuesta in RESPUESTAS.items():
        if clave in pregunta:
            return respuesta
    return RESPUESTA_DEFAULT
```

#### Paso 11 — Integrar el bot en app.py

Agrega esto al final de `app.py`, dentro de un tab o sección separada:

```python
from chatbot import responder

st.divider()
st.subheader("Asistente virtual")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {"role": "assistant", "content": "Hola, soy el asistente del sistema antifraude. ¿En qué puedo ayudarte?"}
    ]

for msg in st.session_state.mensajes:
    st.chat_message(msg["role"]).write(msg["content"])

if pregunta := st.chat_input("Escribe tu pregunta..."):
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    st.chat_message("user").write(pregunta)

    respuesta = responder(pregunta)
    st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
    st.chat_message("assistant").write(respuesta)
```

---

## Integración final — cuando Persona B entregue su modelo

Cuando recibas `models/classification_model.pkl`, haz exactamente esto en `app.py`:

1. Comenta o elimina el bloque `ModeloDummy`
2. Descomenta las líneas de carga real:

```python
regression_model     = joblib.load('models/regression_model.pkl')
classification_model = joblib.load('models/classification_model.pkl')
scaler               = joblib.load('models/scaler.pkl')
```

3. Ajusta el preprocesamiento de `entrada` para usar el mismo scaler de B si es necesario.

---

## Checklist antes de la demo

- [ ] `regression_model.pkl` guardado en `models/`
- [ ] `app.py` corre con `streamlit run app/app.py` sin errores
- [ ] El formulario acepta datos y muestra predicciones
- [ ] El bot responde al menos 5 preguntas distintas
- [ ] Los modelos reales de B están integrados (no el dummy)
- [ ] La app funciona en el computador donde se va a presentar

---

## Cómo ejecutar la app

```bash
# Instalar dependencias
pip install streamlit pandas numpy scikit-learn joblib

# Correr la aplicación
streamlit run app/app.py
```

---

## Rama Git recomendada

```bash
git checkout -b personaC
git add notebooks/persona_C_regression_app_bot.ipynb app/app.py app/chatbot.py models/
git commit -m "feat: regresión, interfaz streamlit y bot conversacional"
git push origin personaC
```
