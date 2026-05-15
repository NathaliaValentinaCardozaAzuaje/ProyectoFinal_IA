"""
app.py — Sistema Inteligente de Detección de Fraude Financiero
==============================================================
Proyecto Final | Introducción a la Inteligencia Artificial — EAFIT 2026-1
Módulos: EDA + Classification + Regression (incluye App + Bot)

Estructura de carpetas esperada:
  proyecto/
  ├── app/
  │   ├── app.py          ← este archivo
  │   └── chatbot.py
  ├── models/
  │   ├── scaler.pkl              ← RobustScaler([Time, Amount]) — 2 features — EDA
  │   ├── regression_model.pkl    ← GradientBoosting/RF regressor — generado por Regression
  │   ├── dnn_regression.pth      ← DNN PyTorch (opcional) — generado por Regression
  │   └── classification_model.pkl← RandomForest classifier (30 features) — generado por Classification
  ├── reports/
  │   ├── regression_metrics.json
  │   └── *.png
  └── data/
      └── creditcard.csv

Ejecución:
  streamlit run app/app.py

NOTA TÉCNICA:
  - scaler.pkl: RobustScaler ajustado sobre [Time, Amount] (2 features) por EDA.
  - REGRESSION: recibe las 30 features con Time y Amount ya escalados.
  - CLASSIFICATION: recibe las mismas 30 features (entrenado con ellas).
  - La app escala Time y Amount con scaler, luego concatena V1-V28 sin escalar adicional.
  - Si los modelos no están disponibles, el modo Demo provee estimaciones ilustrativas.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Carga variables desde .env en la raiz del proyecto (si python-dotenv esta instalado).
if load_dotenv is not None:
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from chatbot import responder, PREGUNTAS_SUGERIDAS, estado_chatbot

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
MODELS_EDA    = os.path.join(BASE_DIR, "models", "eda")
MODELS_CLASS  = os.path.join(BASE_DIR, "models", "classification")
MODELS_REG    = os.path.join(BASE_DIR, "models", "regression")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
REPORTS_EDA   = os.path.join(BASE_DIR, "reports", "eda")
REPORTS_CLASS = os.path.join(BASE_DIR, "reports", "classification")
REPORTS_REG   = os.path.join(BASE_DIR, "reports", "regression")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sistema Antifraude | EAFIT IA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── HEADER ── */
.header-wrap {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e1b4b 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.2rem 3rem;
    margin-bottom: 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.header-wrap::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.header-title {
    font-family: 'Space Mono', monospace;
    color: #e2e8f0;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-sub {
    color: #818cf8;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    letter-spacing: 0.5px;
}
.badge-row { margin-top: 0.8rem; }
.badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin: 0 0.2rem;
}
.badge-eafit  { background: rgba(99,102,241,0.15); border: 1px solid #6366f1; color: #a5b4fc; }
.badge-demo   { background: rgba(245,158,11,0.15);  border: 1px solid #f59e0b; color: #fbbf24; }
.badge-parcial{ background: rgba(249,115,22,0.15);  border: 1px solid #f97316; color: #fb923c; }
.badge-ok     { background: rgba(16,185,129,0.15);  border: 1px solid #10b981; color: #34d399; }

/* ── SEMÁFORO ── */
.risk-card {
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1rem;
}
.risk-low    { background: linear-gradient(135deg,#022c22,#064e3b); border: 2px solid #10b981; }
.risk-medium { background: linear-gradient(135deg,#451a03,#78350f); border: 2px solid #f59e0b; }
.risk-high   { background: linear-gradient(135deg,#450a0a,#7f1d1d); border: 2px solid #ef4444; }
.risk-icon   { font-size: 3.2rem; }
.risk-label  { font-family: 'Space Mono', monospace; font-size: 1.4rem; color: #fff; font-weight: 700; margin-top: 0.4rem; }
.risk-desc   { font-size: 0.9rem; color: rgba(255,255,255,0.75); margin-top: 0.3rem; }

/* ── TARJETAS MÉTRICAS ── */
.metric-card {
    background: linear-gradient(135deg,#0f172a,#1e1b4b);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
}
.metric-val   { font-family:'Space Mono',monospace; font-size: 1.9rem; font-weight: 700; color: #a5b4fc; }
.metric-label { font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── BOTÓN ANALIZAR ── */
.stButton > button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.2px !important;
    transition: box-shadow 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(99,102,241,0.45) !important;
}

/* ── INFO BOX ── */
.info-box {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366f1;
    padding: 0.8rem 1rem;
    border-radius: 0 10px 10px 0;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: #94a3b8;
}

/* ── MODO DEMO BANNER ── */
.demo-banner {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #fbbf24;
    margin-bottom: 1rem;
}

/* ── CHAT ── */
.chat-area {
    background: #0f172a;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 1.2rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────

# Ambos modelos reciben las mismas 30 features: [Time, Amount, V1..V28]
# Time y Amount son escalados con scaler.pkl (RobustScaler, 2 features)
# V1-V28 ya vienen en escala PCA apropiada, no requieren escalado adicional
ALL_FEATURES = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]  # 30 features

# ─────────────────────────────────────────────────────────────────────────────
# MODELO DUMMY (fallback si los .pkl no existen)
# ─────────────────────────────────────────────────────────────────────────────

class _Dummy:
    """Estimaciones ilustrativas para modo Demo."""
    def predict(self, X):
        seed = int(abs(np.asarray(X).sum()) * 997) % 2**31
        rng = np.random.default_rng(seed)
        return rng.uniform(0.05, 0.45, len(X))

    def predict_proba(self, X):
        p = self.predict(X)
        return np.column_stack([1 - p, p])


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cargar_modelos():
    """
    Carga los modelos en orden de prioridad:
      1. scaler.pkl        → RobustScaler de EDA (2 features: Time, Amount)
      2. regression_model.pkl → regresor de Regression
      3. classification_model.pkl → clasificador de Classification
    Retorna (reg, clf, scaler, modo)
    """
    reg    = _Dummy()
    clf    = _Dummy()
    scaler = None
    modo   = "demo"

    # Scaler — buscar en models/eda/ primero, luego raíz
    for _p in [os.path.join(MODELS_EDA, "scaler.pkl"), os.path.join(MODELS_DIR, "scaler.pkl")]:
        if os.path.exists(_p):
            try:
                scaler = joblib.load(_p)
                modo = "parcial"
            except Exception as e:
                st.warning(f"scaler.pkl no se pudo cargar: {e}")
            break

    # Modelo de regresión — buscar en models/regression/ primero
    for _p in [os.path.join(MODELS_REG, "regression_model.pkl"), os.path.join(MODELS_DIR, "regression_model.pkl")]:
        if os.path.exists(_p) and scaler is not None:
            try:
                reg = joblib.load(_p)
            except Exception as e:
                st.warning(f"regression_model.pkl no se pudo cargar: {e}")
            break

    # Modelo de clasificación — buscar en models/classification/ primero
    for _p in [os.path.join(MODELS_CLASS, "classification_model.pkl"), os.path.join(MODELS_DIR, "classification_model.pkl")]:
        if os.path.exists(_p):
            try:
                clf = joblib.load(_p)
                if modo == "parcial":
                    modo = "completo"
            except Exception as e:
                st.warning(f"classification_model.pkl no se pudo cargar: {e}")
            break

    return reg, clf, scaler, modo


reg_model, clf_model, scaler, MODO = cargar_modelos()

# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def predecir(time, amount, v1, v2, v3, v4, v5, v6, v7, v8, v9,
             v10, v11, v12, v13, v14, v15, v16, v17, v18,
             v19, v20, v21, v22, v23, v24, v25, v26, v27, v28):
    """
    Genera risk_score (regresión) y prob_fraude (clasificación).

    Ambos modelos reciben las mismas 30 features en el mismo orden:
    [Time, Amount, V1..V28] con Time y Amount escalados por scaler.pkl.

    scaler.pkl = RobustScaler ajustado sobre [Time, Amount] (2 features).
    La app escala solo esas 2 columnas y deja V1-V28 tal como están.
    """
    def _input_para_estimador(estimator, row_dict, default_cols):
        if hasattr(estimator, "feature_names_in_"):
            cols = list(estimator.feature_names_in_)
            faltantes = [c for c in cols if c not in row_dict]
            if faltantes:
                raise ValueError(
                    f"Faltan features requeridas por el modelo: {faltantes}"
                )
            return pd.DataFrame([[row_dict[c] for c in cols]], columns=cols)
        return np.array([[row_dict[c] for c in default_cols]], dtype=float)

    # Escalar Time y Amount en el formato esperado por el scaler
    if scaler is not None:
        base_raw = {"Time": float(time), "Amount": float(amount)}
        X_scaler = _input_para_estimador(scaler, base_raw, ["Time", "Amount"])
        time_scaled, amount_scaled = scaler.transform(X_scaler)[0]
    else:
        time_scaled, amount_scaled = time, amount

    row = {
        "Time": float(time_scaled), "Amount": float(amount_scaled),
        "V1": float(v1), "V2": float(v2), "V3": float(v3), "V4": float(v4), "V5": float(v5),
        "V6": float(v6), "V7": float(v7), "V8": float(v8), "V9": float(v9), "V10": float(v10),
        "V11": float(v11), "V12": float(v12), "V13": float(v13), "V14": float(v14),
        "V15": float(v15), "V16": float(v16), "V17": float(v17), "V18": float(v18),
        "V19": float(v19), "V20": float(v20), "V21": float(v21), "V22": float(v22),
        "V23": float(v23), "V24": float(v24), "V25": float(v25), "V26": float(v26),
        "V27": float(v27), "V28": float(v28),
    }

    X_reg = _input_para_estimador(reg_model, row, ALL_FEATURES)
    X_clf = _input_para_estimador(clf_model, row, ALL_FEATURES)

    risk_score  = float(np.clip(reg_model.predict(X_reg)[0], 0.0, 1.0))
    prob_fraude = float(clf_model.predict_proba(X_clf)[0][1])

    return risk_score, prob_fraude


def semaforo(score):
    if score < 0.3:
        return "BAJO",  "risk-low",    "✅", "Transacción probablemente legítima"
    elif score < 0.6:
        return "MEDIO", "risk-medium", "⚠️", "Transacción con características inusuales"
    else:
        return "ALTO",  "risk-high",   "🚨", "Transacción sospechosa de fraude"


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

badge_modo = {
    "demo":     '<span class="badge badge-demo">🟡 Modo Demo</span>',
    "parcial":  '<span class="badge badge-parcial">🟠 Modelos parciales</span>',
    "completo": '<span class="badge badge-ok">🟢 Modelos reales</span>',
}.get(MODO, "")

st.markdown(f"""
<div class="header-wrap">
    <p class="header-title">🛡️ Sistema Inteligente de Detección de Fraude</p>
    <p class="header-sub">Machine Learning aplicado a transacciones financieras en tiempo real</p>
    <div class="badge-row">
        <span class="badge badge-eafit">Introducción a la IA — EAFIT 2026-1</span>
        {badge_modo}
    </div>
</div>
""", unsafe_allow_html=True)

if MODO == "demo":
    st.markdown("""
    <div class="demo-banner">
        ⚠️ <strong>Modo Demo activo</strong> — Los modelos reales no se encontraron en <code>models/</code>.
        Ejecuta los notebooks en orden: <strong>EDA → Clasificación → Regresión</strong> para generarlos.
        Las predicciones actuales son ilustrativas.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Analizador", "📊 Dashboard", "🤖 Asistente"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALIZADOR
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Ingresa los datos de la transacción")
    st.markdown(
        '<div class="info-box">💡 Los campos básicos son suficientes para una predicción. '
        'Los parámetros avanzados (variables PCA V1–V28) permiten afinar el análisis.</div>',
        unsafe_allow_html=True
    )

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("**Parámetros básicos**")
        amount = st.number_input(
            "💰 Monto de la transacción (USD)",
            min_value=0.0, max_value=25000.0, value=150.0, step=0.01,
            help="Importe de la transacción"
        )
        time_val = st.number_input(
            "⏱️ Tiempo desde primera transacción (segundos)",
            min_value=0.0, max_value=172800.0, value=50000.0, step=1.0,
            help="Segundos desde la primera transacción del dataset (0–172800)"
        )

        with st.expander("🔧 Parámetros avanzados — Variables PCA (V1–V28)"):
            st.caption("Las variables más discriminativas son V1, V2, V3, V4, V10, V12, V14, V16, V17. "
                       "Deja en 0.0 si no dispones de los valores.")
            c1, c2, c3 = st.columns(3)
            with c1:
                v1  = st.slider("V1",  -10.0, 10.0, 0.0, 0.1)
                v2  = st.slider("V2",  -10.0, 10.0, 0.0, 0.1)
                v3  = st.slider("V3",  -10.0, 10.0, 0.0, 0.1)
                v4  = st.slider("V4",  -10.0, 10.0, 0.0, 0.1)
                v5  = st.slider("V5",  -10.0, 10.0, 0.0, 0.1)
                v6  = st.slider("V6",  -10.0, 10.0, 0.0, 0.1)
                v7  = st.slider("V7",  -10.0, 10.0, 0.0, 0.1)
                v8  = st.slider("V8",  -10.0, 10.0, 0.0, 0.1)
                v9  = st.slider("V9",  -10.0, 10.0, 0.0, 0.1)
                v10 = st.slider("V10", -10.0, 10.0, 0.0, 0.1)
            with c2:
                v11 = st.slider("V11", -10.0, 10.0, 0.0, 0.1)
                v12 = st.slider("V12", -10.0, 10.0, 0.0, 0.1)
                v13 = st.slider("V13", -10.0, 10.0, 0.0, 0.1)
                v14 = st.slider("V14", -10.0, 10.0, 0.0, 0.1)
                v15 = st.slider("V15", -10.0, 10.0, 0.0, 0.1)
                v16 = st.slider("V16", -10.0, 10.0, 0.0, 0.1)
                v17 = st.slider("V17", -10.0, 10.0, 0.0, 0.1)
                v18 = st.slider("V18", -10.0, 10.0, 0.0, 0.1)
                v19 = st.slider("V19", -10.0, 10.0, 0.0, 0.1)
            with c3:
                v20 = st.slider("V20", -10.0, 10.0, 0.0, 0.1)
                v21 = st.slider("V21", -10.0, 10.0, 0.0, 0.1)
                v22 = st.slider("V22", -10.0, 10.0, 0.0, 0.1)
                v23 = st.slider("V23", -10.0, 10.0, 0.0, 0.1)
                v24 = st.slider("V24", -10.0, 10.0, 0.0, 0.1)
                v25 = st.slider("V25", -10.0, 10.0, 0.0, 0.1)
                v26 = st.slider("V26", -10.0, 10.0, 0.0, 0.1)
                v27 = st.slider("V27", -10.0, 10.0, 0.0, 0.1)
                v28 = st.slider("V28", -10.0, 10.0, 0.0, 0.1)

        analizar = st.button("🔍 Analizar transacción", width='stretch')

        # Resumen de valores ingresados
        st.markdown("**Resumen de valores ingresados**")
        resumen = {
            "Variable": ["Time","Amount","V1","V2","V3","V4","V5","V6","V7","V8","V9",
                         "V10","V11","V12","V13","V14","V15","V16","V17","V18",
                         "V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"],
            "Valor":    [time_val, amount,
                         v1,v2,v3,v4,v5,v6,v7,v8,v9,
                         v10,v11,v12,v13,v14,v15,v16,v17,v18,
                         v19,v20,v21,v22,v23,v24,v25,v26,v27,v28]
        }
        st.dataframe(pd.DataFrame(resumen), width='stretch', hide_index=True)

    with col_result:
        st.markdown("**Resultado del análisis**")

        if analizar:
            with st.spinner("Analizando transacción con los modelos de ML..."):
                risk_score, prob_fraude = predecir(
                    time_val, amount,
                    v1, v2, v3, v4, v5, v6, v7, v8, v9,
                    v10, v11, v12, v13, v14, v15, v16, v17, v18,
                    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28
                )

            nivel, css, icon, desc = semaforo(risk_score)

            # Semáforo
            st.markdown(f"""
            <div class="risk-card {css}">
                <div class="risk-icon">{icon}</div>
                <div class="risk-label">RIESGO {nivel}</div>
                <div class="risk-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Métricas numéricas
            m1, m2 = st.columns(2)
            with m1:
                st.metric("📊 Risk Score (regresión)", f"{risk_score:.4f}")
                st.progress(risk_score, text=f"{risk_score*100:.1f}%")
            with m2:
                st.metric("🎯 Prob. Fraude (clasificación)", f"{prob_fraude*100:.1f}%")
                st.progress(min(prob_fraude, 1.0), text=f"{prob_fraude*100:.1f}%")

            # Interpretación
            st.markdown("---")
            st.markdown("**📝 Interpretación**")
            if nivel == "ALTO":
                st.error(
                    f"⚠️ La transacción de **${amount:,.2f}** presenta características de alto riesgo "
                    f"(risk_score: **{risk_score:.4f}**). Se recomienda bloqueo preventivo y verificación manual."
                )
            elif nivel == "MEDIO":
                st.warning(
                    f"📋 La transacción de **${amount:,.2f}** muestra algunas anomalías "
                    f"(risk_score: **{risk_score:.4f}**). Se sugiere monitoreo adicional."
                )
            else:
                st.success(
                    f"✅ La transacción de **${amount:,.2f}** parece legítima "
                    f"(risk_score: **{risk_score:.4f}**). Sin señales de alerta significativas."
                )

            # Guardar en historial
            if "historial" not in st.session_state:
                st.session_state.historial = []
            st.session_state.historial.append({
                "Amount ($)":       round(amount, 2),
                "Time (s)":        int(time_val),
                "Risk Score":      round(risk_score, 4),
                "Prob. Fraude (%)": round(prob_fraude * 100, 1),
                "Nivel":           nivel,
            })

        else:
            st.markdown("""
            <div style="padding:3rem;text-align:center;color:#475569;
                        border:2px dashed #1e293b;border-radius:14px;margin-top:1rem;">
                <div style="font-size:3rem">🔍</div>
                <p style="margin-top:0.5rem;font-size:0.95rem">
                    Completa el formulario y presiona<br>
                    <strong style="color:#818cf8">Analizar transacción</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 📊 Dashboard Integrado del Proyecto")

    dash_eda, dash_clf, dash_reg, dash_hist = st.tabs([
        "🔬 EDA", "🎯 Clasificación", "📈 Regresión", "📋 Historial",
    ])

    # ══ EDA ══
    with dash_eda:
        st.markdown("#### Análisis Exploratorio de Datos")
        st.markdown(
            "Exploración del dataset **Credit Card Fraud Detection** "
            "(284,807 transacciones · 0.17% fraude · V1–V28 anonimizadas con PCA)."
        )
        EDA_IMGS = [
            ("distribucion_fraudes.png",       "Distribución de Clases"),
            ("correlaciones_class.png",         "Correlación con Fraude"),
            ("variables_criticas_boxplot.png",  "Boxplots Variables PCA Críticas"),
            ("distribucion_time.png",           "Distribución Temporal por Clase"),
            ("balance_smote.png",               "Balance de Clases tras SMOTE"),
            ("analisis_amount.png",             "Análisis de Montos por Clase"),
            ("comparacion_montos.png",          "Comparación de Montos"),
            ("distribucion_montos.png",         "Distribución de Montos"),
            ("heatmap_correlaciones.png",       "Heatmap de Correlaciones"),
            ("boxplot_montos.png",              "Boxplot de Montos"),
            ("analisis_temporal.png",           "Análisis Temporal"),
            ("pca_features_por_clase.png",      "Variables PCA por Clase"),
            ("risk_score_distribucion.png",     "Distribución del Risk Score"),
            ("risk_score_analisis.png",         "Análisis del Risk Score"),
        ]
        eda_disp = [(f, t) for f, t in EDA_IMGS
                    if os.path.exists(os.path.join(REPORTS_EDA, f))]
        if eda_disp:
            g = st.columns(2)
            for i, (fn, tt) in enumerate(eda_disp):
                with g[i % 2]:
                    st.markdown(f"**{tt}**")
                    st.image(os.path.join(REPORTS_EDA, fn), width='stretch')
                    st.markdown("")
        else:
            st.info("Ejecuta `notebooks/eda.ipynb` para generar estas gráficas.")

    # ══ CLASIFICACIÓN ══
    with dash_clf:
        st.markdown("#### Modelo de Clasificación — Random Forest")
        st.markdown(
            "Clasifica transacciones como **fraude (1)** o **legítima (0)**. "
            "Entrenado con SMOTE · métrica principal: **F1-score** y **ROC-AUC**."
        )
        clf_loaded = not isinstance(clf_model, _Dummy)
        st.markdown(f"**Estado del modelo:** {'✅ Cargado' if clf_loaded else '🟡 Modo Demo'}")
        st.markdown("")
        CLF_IMGS = [
            ("comparacion_modelos_clasificacion.png", "Comparación de Modelos (F1, Precision, Recall)"),
            ("matrices_confusion.png",                "Matrices de Confusión"),
            ("curvas_roc.png",                        "Curvas ROC por Modelo"),
        ]
        clf_disp = [(f, t) for f, t in CLF_IMGS
                    if os.path.exists(os.path.join(REPORTS_CLASS, f))]
        if clf_disp:
            g = st.columns(min(2, len(clf_disp)))
            for i, (fn, tt) in enumerate(clf_disp):
                with g[i % 2]:
                    st.markdown(f"**{tt}**")
                    st.image(os.path.join(REPORTS_CLASS, fn), width='stretch')
        else:
            st.info("Ejecuta `notebooks/classification.ipynb` para generar estas gráficas.")

    # ══ REGRESIÓN ══
    with dash_reg:
        st.markdown("#### Modelo de Regresión — DNN / Gradient Boosting")
        st.markdown(
            "Predice el **risk_score continuo** (0–1). "
            "DNN: arquitectura 512→256→128→64→32→16→1 con WeightedMSE (15× en fraudes). "
            "Métrica principal: **R²** y **RMSE**."
        )
        reg_loaded = not isinstance(reg_model, _Dummy)
        st.markdown(f"**Estado del modelo:** {'✅ Cargado' if reg_loaded else '🟡 Modo Demo'}")
        for _mp in [os.path.join(REPORTS_REG, "regression_metrics.json"),
                    os.path.join(REPORTS_DIR, "regression_metrics.json")]:
            if os.path.exists(_mp):
                with open(_mp) as f:
                    metricas = json.load(f)
                mejor = metricas.get("mejor_modelo", {})
                st.markdown("")
                c1, c2, c3, c4 = st.columns(4)
                for col, (k, label) in zip([c1, c2, c3, c4], [
                    ("nombre", "Mejor Modelo"), ("r2", "R²"),
                    ("rmse", "RMSE"), ("mae", "MAE"),
                ]):
                    val = mejor.get(k, "—")
                    if isinstance(val, float): val = f"{val:.4f}"
                    col.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                if "todos" in metricas:
                    st.markdown("")
                    st.markdown("**Comparación de todos los modelos**")
                    df_comp = pd.DataFrame(metricas["todos"]).T.rename(
                        columns={"r2": "R²", "rmse": "RMSE", "mae": "MAE"}
                    )
                    st.dataframe(
                        df_comp.style.highlight_max(subset=["R²"], color="#064e3b")
                                     .highlight_min(subset=["RMSE", "MAE"], color="#064e3b"),
                        width='stretch'
                    )
                break
        else:
            st.info("Ejecuta `notebooks/regression.ipynb` para generar las métricas.")
        st.markdown("")
        REG_IMGS = [
            ("comparacion_modelos_regresion.png", "Comparación de Modelos de Regresión"),
            ("correlaciones_riskscore.png",        "Correlaciones con Risk Score"),
            ("importancia_variables.png",          "Importancia de Variables"),
            ("dnn_learning_curves.png",            "Curvas de Aprendizaje DNN"),
            ("dnn_analisis_errores.png",           "Análisis de Errores DNN"),
            ("heatmap_correlacion.png",            "Heatmap de Correlación"),
        ]
        reg_disp = [(f, t) for f, t in REG_IMGS
                    if os.path.exists(os.path.join(REPORTS_REG, f))]
        if reg_disp:
            g = st.columns(2)
            for i, (fn, tt) in enumerate(reg_disp):
                with g[i % 2]:
                    st.markdown(f"**{tt}**")
                    st.image(os.path.join(REPORTS_REG, fn), width='stretch')
                    st.markdown("")
        else:
            st.info("Ejecuta `notebooks/regression.ipynb` para generar estas gráficas.")

    # ══ HISTORIAL ══
    with dash_hist:
        st.markdown("#### Historial de análisis — sesión actual")
        if "historial" in st.session_state and st.session_state.historial:
            df_hist = pd.DataFrame(st.session_state.historial)
            df_hist.index = range(1, len(df_hist) + 1)

            def colorear_nivel(val):
                if val == "ALTO":  return "color: #ef4444; font-weight: bold"
                if val == "MEDIO": return "color: #f59e0b; font-weight: bold"
                return "color: #10b981; font-weight: bold"

            st.dataframe(df_hist.style.map(colorear_nivel, subset=["Nivel"]), width='stretch')
            total = len(df_hist)
            altos = (df_hist["Nivel"] == "ALTO").sum()
            medios = (df_hist["Nivel"] == "MEDIO").sum()
            bajos = (df_hist["Nivel"] == "BAJO").sum()
            s1, s2, s3, s4 = st.columns(4)
            for col, (v, l) in zip([s1, s2, s3, s4], [
                (total, "Total"), (altos, "🔴 Alto"),
                (medios, "🟡 Medio"), (bajos, "🟢 Bajo"),
            ]):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{v}</div>
                    <div class="metric-label">{l}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("")
            if st.button("🗑️ Limpiar historial", key="clear_hist_tab"):
                st.session_state.historial = []
                st.rerun()
        else:
            st.caption("No hay transacciones analizadas. Ve al **🔍 Analizador** para comenzar.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ASISTENTE CONVERSACIONAL
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🤖 Asistente Virtual Antifraude")
    st.caption(
        "Pregunta sobre fraude financiero, los modelos de ML, cómo interpretar resultados, "
        "qué son SMOTE o WeightedMSE, o cómo usar la aplicación."
    )
    st.caption(f"Modo del chatbot: **{estado_chatbot()}**")

    # Inicializar historial
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [
            {
                "role": "assistant",
                "content": (
                    "👋 ¡Hola! Soy el asistente del **Sistema Antifraude**. "
                    "Puedo explicarte cómo funcionan los modelos (DNN, Random Forest, Gradient Boosting), "
                    "qué significan las métricas y los resultados, y responder sobre fraude financiero.\n\n"
                    "¿En qué puedo ayudarte?"
                )
            }
        ]

    # Sugerencias iniciales
    if len(st.session_state.mensajes) == 1:
        st.markdown("**💡 Preguntas frecuentes:**")
        cols = st.columns(len(PREGUNTAS_SUGERIDAS))
        for i, (col, preg) in enumerate(zip(cols, PREGUNTAS_SUGERIDAS)):
            with col:
                if st.button(preg, key=f"sug_{i}", width='stretch'):
                    st.session_state.mensajes.append({"role": "user", "content": preg})
                    st.session_state.mensajes.append({"role": "assistant", "content": responder(preg)})
                    st.rerun()
        st.markdown("---")

    # Historial de mensajes
    for msg in st.session_state.mensajes:
        avatar = "🛡️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input del usuario
    if pregunta := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user", avatar="👤"):
            st.markdown(pregunta)

        respuesta = responder(pregunta)
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant", avatar="🛡️"):
            st.markdown(respuesta)

    # Limpiar chat
    if len(st.session_state.mensajes) > 1:
        st.markdown("---")
        if st.button("🗑️ Limpiar conversación", key="clear_chat"):
            st.session_state.mensajes = [st.session_state.mensajes[0]]
            st.rerun()
