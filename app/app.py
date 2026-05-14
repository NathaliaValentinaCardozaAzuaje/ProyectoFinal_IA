"""
app.py — Interfaz Streamlit para el Sistema de Detección de Fraude Financiero
=============================================================================
Persona C | Proyecto Final IA — EAFIT 2026-1

Estructura:
  Tab 1: 🔍 Analizador     — formulario + predicciones + semáforo de riesgo
  Tab 2: 📊 Dashboard      — métricas del modelo + gráficas de referencia
  Tab 3: 🤖 Asistente      — chatbot conversacional integrado

Ejecución:
  streamlit run app/app.py
"""

import sys
import os

# Asegurar que Python encuentre chatbot.py en la misma carpeta
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from chatbot import responder, PREGUNTAS_SUGERIDAS

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
# CSS PERSONALIZADO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Fuente moderna */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header principal */
.header-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.header-title {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-subtitle {
    color: #a78bfa;
    font-size: 1rem;
    margin-top: 0.3rem;
}
.header-badge {
    display: inline-block;
    background: rgba(167,139,250,0.2);
    border: 1px solid #a78bfa;
    color: #a78bfa;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    margin-top: 0.5rem;
}

/* Tarjetas de métricas */
.metric-card {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-card-label {
    font-size: 0.85rem;
    color: #c4b5fd;
    margin-top: 0.2rem;
}

/* Semáforo de riesgo */
.risk-low    { background: linear-gradient(135deg, #064e3b, #065f46); border: 2px solid #10b981; border-radius: 12px; padding: 1.5rem; text-align: center; }
.risk-medium { background: linear-gradient(135deg, #78350f, #92400e); border: 2px solid #f59e0b; border-radius: 12px; padding: 1.5rem; text-align: center; }
.risk-high   { background: linear-gradient(135deg, #7f1d1d, #991b1b); border: 2px solid #ef4444; border-radius: 12px; padding: 1.5rem; text-align: center; }

.risk-emoji  { font-size: 3rem; }
.risk-nivel  { font-size: 1.5rem; font-weight: 700; color: white; margin-top: 0.5rem; }
.risk-desc   { font-size: 0.9rem; color: rgba(255,255,255,0.8); margin-top: 0.3rem; }

/* Botón analizar */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: transform 0.1s ease, box-shadow 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.4) !important;
}

/* Sección de chat */
.chat-sugerencia {
    background: rgba(79,70,229,0.1);
    border: 1px solid #4f46e5;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    margin: 0.3rem;
    font-size: 0.85rem;
    cursor: pointer;
    display: inline-block;
    color: #a78bfa;
}

/* Info box */
.info-box {
    background: rgba(79,70,229,0.1);
    border-left: 4px solid #4f46e5;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
}

/* Modo demo badge */
.demo-badge {
    background: rgba(245,158,11,0.15);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELOS (con fallback elegante)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class _ModeloDummy:
    """Modelo simulado para usar mientras los modelos reales están en desarrollo."""
    def __init__(self, seed=42):
        self._seed = seed

    def predict(self, X):
        rng = np.random.default_rng(int(abs(np.array(X).sum()) * 1000) % 2**31)
        return rng.uniform(0.05, 0.45, len(X))

    def predict_proba(self, X):
        p = self.predict(X)
        return np.column_stack([1 - p, p])


@st.cache_resource(show_spinner=False)
def cargar_modelos():
    reg_path = os.path.join(MODELS_DIR, "regression_model.pkl")
    clf_path = os.path.join(MODELS_DIR, "classification_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    modo = "demo"
    reg, clf, scaler = _ModeloDummy(), _ModeloDummy(), None

    if os.path.exists(reg_path) and os.path.exists(scaler_path):
        try:
            reg    = joblib.load(reg_path)
            scaler = joblib.load(scaler_path)
            modo   = "parcial"
        except Exception:
            pass

    if os.path.exists(clf_path):
        try:
            clf  = joblib.load(clf_path)
            modo = "completo" if modo == "parcial" else modo
        except Exception:
            pass

    return reg, clf, scaler, modo


reg_model, clf_model, scaler, MODO = cargar_modelos()

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4',
            'V10', 'V12', 'V14', 'V16', 'V17']


def predecir(time, amount, v1, v2, v3, v4, v10, v12, v14, v16, v17):
    entrada = np.array([[time, amount, v1, v2, v3, v4, v10, v12, v14, v16, v17]])

    if scaler is not None:
        entrada_scaled = scaler.transform(entrada)
    else:
        entrada_scaled = entrada

    risk_score   = float(np.clip(reg_model.predict(entrada_scaled)[0], 0, 1))
    prob_fraude  = float(clf_model.predict_proba(entrada)[0][1])

    return risk_score, prob_fraude


def clasificar_riesgo(score):
    if score < 0.3:
        return "BAJO", "risk-low", "✅", "Transacción probablemente legítima"
    elif score < 0.6:
        return "MEDIO", "risk-medium", "⚠️", "Transacción con características inusuales"
    else:
        return "ALTO", "risk-high", "🚨", "Transacción sospechosa de fraude"


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

modo_label = {"demo": "🟡 Modo Demo", "parcial": "🟠 Modelos parciales", "completo": "🟢 Modelos reales"}

st.markdown(f"""
<div class="header-container">
    <p class="header-title">🛡️ Sistema Inteligente de Detección de Fraude</p>
    <p class="header-subtitle">Machine Learning aplicado a transacciones financieras</p>
    <span class="header-badge">Introducción a la IA — EAFIT 2026-1 · Persona C</span>
    &nbsp;&nbsp;
    <span class="demo-badge">{modo_label.get(MODO, MODO)}</span>
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
        '<div class="info-box">💡 Los campos básicos son suficientes para obtener una predicción. '
        'Los parámetros avanzados (variables PCA) permiten afinar el análisis si dispones de ellos.</div>',
        unsafe_allow_html=True
    )

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        # Campos principales
        st.markdown("**Parámetros básicos**")
        amount = st.number_input(
            "💰 Monto de la transacción (USD)",
            min_value=0.0, max_value=25000.0, value=150.0, step=0.01,
            help="Importe de la transacción en dólares estadounidenses"
        )
        time = st.number_input(
            "⏱️ Tiempo desde primera transacción (segundos)",
            min_value=0.0, max_value=172800.0, value=50000.0, step=1.0,
            help="Segundos transcurridos desde la primera transacción del dataset"
        )

        # Parámetros avanzados
        with st.expander("🔧 Parámetros avanzados (Variables PCA V1–V17)"):
            st.caption("Dejar en 0.0 si no se dispone de los valores originales.")
            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                v1  = st.slider("V1",  -5.0, 5.0, 0.0, 0.1)
                v2  = st.slider("V2",  -5.0, 5.0, 0.0, 0.1)
                v3  = st.slider("V3",  -5.0, 5.0, 0.0, 0.1)
                v4  = st.slider("V4",  -5.0, 5.0, 0.0, 0.1)
                v10 = st.slider("V10", -5.0, 5.0, 0.0, 0.1)
            with adv_col2:
                v12 = st.slider("V12", -5.0, 5.0, 0.0, 0.1)
                v14 = st.slider("V14", -5.0, 5.0, 0.0, 0.1)
                v16 = st.slider("V16", -5.0, 5.0, 0.0, 0.1)
                v17 = st.slider("V17", -5.0, 5.0, 0.0, 0.1)

        analizar = st.button("🔍 Analizar transacción", use_container_width=True)

        # Tabla de valores ingresados
        st.markdown("**Resumen de valores ingresados**")
        df_entrada = pd.DataFrame({
            "Variable": FEATURES,
            "Valor": [time, amount, v1, v2, v3, v4, v10, v12, v14, v16, v17]
        })
        st.dataframe(df_entrada, use_container_width=True, hide_index=True)

    with col_result:
        st.markdown("**Resultado del análisis**")

        if analizar:
            with st.spinner("Analizando transacción..."):
                risk_score, prob_fraude = predecir(
                    time, amount, v1, v2, v3, v4, v10, v12, v14, v16, v17
                )

            nivel, css_class, emoji, desc = clasificar_riesgo(risk_score)

            # Semáforo visual
            st.markdown(f"""
            <div class="{css_class}">
                <div class="risk-emoji">{emoji}</div>
                <div class="risk-nivel">RIESGO {nivel}</div>
                <div class="risk-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Métricas numéricas
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label="📊 Risk Score",
                    value=f"{risk_score:.4f}",
                    help="Puntaje continuo de riesgo (0 = bajo, 1 = alto)"
                )
                st.progress(risk_score, text=f"{risk_score*100:.1f}%")

            with m2:
                st.metric(
                    label="🎯 Prob. de Fraude",
                    value=f"{prob_fraude*100:.1f}%",
                    help="Probabilidad estimada de que sea fraude"
                )
                st.progress(prob_fraude, text=f"{prob_fraude*100:.1f}%")

            # Interpretación textual
            st.markdown("---")
            st.markdown("**📝 Interpretación**")
            if nivel == "ALTO":
                st.error(
                    f"⚠️ La transacción de **${amount:,.2f}** presenta características de alto riesgo. "
                    f"Puntaje: **{risk_score:.4f}**. Se recomienda bloqueo preventivo y verificación manual."
                )
            elif nivel == "MEDIO":
                st.warning(
                    f"📋 La transacción de **${amount:,.2f}** muestra algunas anomalías. "
                    f"Puntaje: **{risk_score:.4f}**. Se sugiere monitoreo adicional."
                )
            else:
                st.success(
                    f"✅ La transacción de **${amount:,.2f}** parece legítima. "
                    f"Puntaje: **{risk_score:.4f}**. Sin señales de alerta significativas."
                )

            # Guardar en session state para el dashboard
            if "historial" not in st.session_state:
                st.session_state.historial = []
            st.session_state.historial.append({
                "Amount": amount, "Time": time,
                "Risk Score": round(risk_score, 4),
                "Prob. Fraude %": round(prob_fraude * 100, 1),
                "Nivel": nivel,
            })

        else:
            st.markdown("""
            <div style="padding:3rem; text-align:center; color:#6b7280; border: 2px dashed #374151; border-radius:12px;">
                <p style="font-size:3rem; margin:0">🔍</p>
                <p style="margin-top:0.5rem">Completa el formulario y presiona<br><strong>Analizar transacción</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 📊 Dashboard del Modelo")

    # Métricas del modelo de regresión (se actualizarán con valores reales al ejecutar el notebook)
    st.markdown("#### Métricas del Modelo de Regresión")

    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    metrics_path = os.path.join(REPORTS_DIR, "regression_metrics.json")

    import json
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metricas = json.load(f)
        mejor = metricas.get("mejor_modelo", {})
        c1, c2, c3, c4 = st.columns(4)
        for col, (k, label) in zip([c1, c2, c3, c4], [
            ("nombre", "Mejor Modelo"),
            ("r2",     "R²"),
            ("rmse",   "RMSE"),
            ("mae",    "MAE"),
        ]):
            val = mejor.get(k, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-value">{val}</div>
                <div class="metric-card-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Ejecuta el notebook para generar las métricas reales del modelo.")
        c1, c2, c3, c4 = st.columns(4)
        placeholder_data = [
            ("Gradient Boosting", "Mejor Modelo"),
            ("—", "R²"),
            ("—", "RMSE"),
            ("—", "MAE"),
        ]
        for col, (val, label) in zip([c1, c2, c3, c4], placeholder_data):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-value">{val}</div>
                <div class="metric-card-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Gráficas guardadas por el notebook
    st.markdown("#### Visualizaciones del Análisis")

    graficas = [
        ("eda_distribucion.png",           "EDA: Distribución de Clases y Montos"),
        ("heatmap_correlaciones.png",      "Heatmap de Correlaciones Top 15"),
        ("risk_score_analisis.png",        "Análisis del Risk Score"),
        ("comparacion_modelos.png",        "Comparación de Modelos (Clásicos vs DNN)"),
        ("dnn_curvas_aprendizaje.png",     "Curvas de Aprendizaje DNN"),
        ("analisis_errores.png",           "Análisis de Errores del Mejor Modelo"),
        ("importancia_variables.png",      "Importancia de Variables (Random Forest)")
    ]

    graficas_disponibles = [
        (fname, titulo)
        for fname, titulo in graficas
        if os.path.exists(os.path.join(REPORTS_DIR, fname))
    ]

    if graficas_disponibles:
        g1, g2 = st.columns(2)
        for i, (fname, titulo) in enumerate(graficas_disponibles):
            col = g1 if i % 2 == 0 else g2
            with col:
                st.markdown(f"**{titulo}**")
                st.image(os.path.join(REPORTS_DIR, fname), use_container_width=True)
    else:
        st.info("ℹ️ Ejecuta el notebook `persona_C_regression_app_bot.ipynb` para generar las gráficas.")

    st.markdown("---")

    # Historial de transacciones analizadas en esta sesión
    st.markdown("#### Historial de análisis (sesión actual)")
    if "historial" in st.session_state and st.session_state.historial:
        df_hist = pd.DataFrame(st.session_state.historial)
        df_hist.index = range(1, len(df_hist) + 1)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.caption("No hay transacciones analizadas en esta sesión. Ve al **Analizador** para comenzar.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ASISTENTE CONVERSACIONAL
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🤖 Asistente Virtual Antifraude")
    st.caption(
        "Pregunta sobre fraude financiero, cómo interpretar los resultados, "
        "qué hacen los modelos, o cómo usar la aplicación."
    )

    # Inicializar historial de chat
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [
            {
                "role": "assistant",
                "content": (
                    "👋 ¡Hola! Soy el asistente del **Sistema Antifraude**. "
                    "Puedo ayudarte a entender los resultados del análisis, explicar "
                    "cómo funcionan los modelos y responder preguntas sobre fraude financiero.\n\n"
                    "¿En qué puedo ayudarte hoy?"
                )
            }
        ]

    # Sugerencias de preguntas (solo al inicio)
    if len(st.session_state.mensajes) == 1:
        st.markdown("**💡 Preguntas frecuentes:**")
        cols_sug = st.columns(len(PREGUNTAS_SUGERIDAS))
        for i, (col, pregunta) in enumerate(zip(cols_sug, PREGUNTAS_SUGERIDAS)):
            with col:
                if st.button(pregunta, key=f"sug_{i}", use_container_width=True):
                    # Agregar pregunta sugerida como si el usuario la hubiera escrito
                    st.session_state.mensajes.append({"role": "user", "content": pregunta})
                    respuesta = responder(pregunta)
                    st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
                    st.rerun()
        st.markdown("---")

    # Mostrar historial de mensajes
    for msg in st.session_state.mensajes:
        with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Input del usuario
    if pregunta_usuario := st.chat_input("Escribe tu pregunta aquí..."):
        # Mostrar mensaje del usuario
        st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
        with st.chat_message("user", avatar="👤"):
            st.markdown(pregunta_usuario)

        # Generar y mostrar respuesta
        respuesta_bot = responder(pregunta_usuario)
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta_bot})
        with st.chat_message("assistant", avatar="🛡️"):
            st.markdown(respuesta_bot)

    # Botón para limpiar el chat
    if len(st.session_state.mensajes) > 1:
        st.markdown("---")
        if st.button("🗑️ Limpiar conversación", key="clear_chat"):
            st.session_state.mensajes = [st.session_state.mensajes[0]]
            st.rerun()
