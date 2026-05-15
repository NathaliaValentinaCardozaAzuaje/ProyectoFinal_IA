"""
chatbot.py — Bot conversacional para el Sistema Antifraude
Cubre los 3 módulos: EDA, Classification y Regression/App.
Usa keyword matching con normalización de tildes y prioriza claves más largas.
"""
import json
import os
import re
from pathlib import Path
from urllib import error, request

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# BASE DE CONOCIMIENTO
# ─────────────────────────────────────────────────────────────────────────────

RESPUESTAS = {
    # ── NIVELES DE RIESGO ────────────────────────────────────────────────────
    "riesgo alto": (
        "🚨 **Riesgo Alto** (score > 0.6): el modelo detectó patrones muy similares a fraudes "
        "históricos del dataset. No garantiza fraude, pero se recomienda **bloqueo preventivo** "
        "y verificación manual antes de aprobar la transacción."
    ),
    "riesgo medio": (
        "⚠️ **Riesgo Medio** (score 0.3–0.6): la transacción tiene algunas características "
        "inusuales (monto atípico o variables PCA fuera de rango), pero no suficientes para "
        "catalogarla como fraude. Se sugiere monitoreo adicional."
    ),
    "riesgo bajo": (
        "✅ **Riesgo Bajo** (score < 0.3): la transacción tiene características similares a "
        "operaciones legítimas. El sistema la considera segura, aunque ningún modelo es 100% infalible."
    ),

    # ── CONCEPTOS CORE ───────────────────────────────────────────────────────
    "risk score": (
        "📊 El **risk_score** es un valor continuo entre 0.0 y 1.0 que combina tres señales:\n\n"
        "- **60%** → si la transacción es fraude confirmado (`Class`)\n"
        "- **30%** → monto normalizado (`Amount`): montos atípicos elevan el riesgo\n"
        "- **10%** → magnitud promedio de las variables PCA críticas (V1, V2, V4, V10, V12, V14, V16, V17)\n\n"
        "El resultado se normaliza a [0, 1]:\n"
        "🟢 0.0–0.3 → Bajo · 🟡 0.3–0.6 → Medio · 🔴 0.6–1.0 → Alto"
    ),
    "puntaje de riesgo": (
        "📊 El **puntaje de riesgo** es el mismo que el risk_score: un número entre 0 y 1. "
        "Cuanto más cercano a 1, mayor la probabilidad de que la transacción sea fraudulenta. "
        "Es producido por el modelo de regresión (DNN o Gradient Boosting según disponibilidad)."
    ),

    # ── FRAUDE ───────────────────────────────────────────────────────────────
    "fraude": (
        "🔍 El sistema detecta fraude analizando patrones en múltiples variables:\n\n"
        "- **Amount**: monto de la transacción\n"
        "- **Time**: segundos desde el inicio del dataset\n"
        "- **V1–V28**: 28 componentes PCA del comportamiento real de la tarjeta (anonimizados)\n\n"
        "Cuando esos patrones coinciden con fraudes históricos del dataset de "
        "284,807 transacciones reales, se emite una alerta. Solo el **0.17%** de las "
        "transacciones son fraude, lo que hace el problema muy desafiante."
    ),

    # ── SISTEMA ──────────────────────────────────────────────────────────────
    "que hace": (
        "🤖 Este sistema analiza transacciones financieras con Machine Learning:\n\n"
        "1. **EDA**: exploración y preparación del dataset, escalado, SMOTE y creación del risk_score\n"
        "2. **Classification**: Random Forest que decide si la transacción es fraude (1) o no (0)\n"
        "3. **Regression**: DNN (Red Neuronal) o Gradient Boosting que predice el risk_score continuo\n\n"
        "Todo se integra en esta app Streamlit con chatbot incluido."
    ),
    "sistema": (
        "🏦 El **Sistema Antifraude** evalúa transacciones bancarias en tiempo real con IA. "
        "Ingresa los datos de una transacción y obtén en segundos:\n\n"
        "- Un **risk_score** continuo (modelo de regresión)\n"
        "- Una **probabilidad de fraude** (modelo de clasificación)\n"
        "- Un **semáforo visual** de riesgo (Bajo / Medio / Alto)\n\n"
        "Usa la pestaña **🔍 Analizador** para probarlo."
    ),

    # ── MODELOS ──────────────────────────────────────────────────────────────
    "modelo": (
        "⚙️ Usamos **dos modelos complementarios**:\n\n"
        "- **Regresión (DNN / Gradient Boosting):** predice el puntaje continuo de riesgo (0–1). "
        "La DNN usa arquitectura 512→256→128→64→32→16→1 con BatchNorm, Dropout y WeightedMSE.\n"
        "- **Clasificación (Random Forest):** decide si es fraude (1) o no (0), entrenado con SMOTE.\n\n"
        "Ambos entrenados con el dataset 'Credit Card Fraud Detection' de Kaggle (ULB, Bruselas)."
    ),
    "red neuronal": (
        "🧠 La **Red Neuronal Profunda (DNN)** es el modelo principal de regresión:\n\n"
        "- Arquitectura: 512 → 256 → 128 → 64 → 32 → 16 → 1 neuronas\n"
        "- Activación: ReLU en capas ocultas, Sigmoid en salida (garantiza [0,1])\n"
        "- Regularización: BatchNorm + Dropout decreciente (0.35 → 0.05)\n"
        "- Optimizador: AdamW con weight_decay\n"
        "- Pérdida: WeightedMSE (penaliza 15× los errores en fraudes)\n"
        "- Entrenada con Early Stopping para evitar overfitting"
    ),
    "regresion": (
        "📈 El **modelo de regresión** predice el risk_score continuo (0–1). "
        "Se compararon Linear Regression, Random Forest, Gradient Boosting y una **DNN (PyTorch)**. "
        "El mejor modelo (mayor R² y menor RMSE) se selecciona automáticamente. "
        "La DNN usa WeightedMSE que penaliza 15× los errores en transacciones de fraude."
    ),
    "regression": (
        "📈 **Regression** predice el risk_score continuo (0–1). "
        "Se compararon Linear Regression, Random Forest, Gradient Boosting y una DNN (PyTorch), "
        "seleccionando el mejor por mayor R² y menor RMSE."
    ),
    "clasificacion": (
        "🎯 El **modelo de clasificación** predice si la transacción es fraude (1) o legítima (0). "
        "Se compararon Logistic Regression, Decision Tree y **Random Forest** (ganador por F1-score). "
        "Fue entrenado con SMOTE para manejar el desbalance extremo del 0.17% de fraudes."
    ),
    "classification": (
        "🎯 **Classification** predice si la transacción es fraude (1) o legítima (0). "
        "El modelo principal es Random Forest, evaluado con F1-score y ROC-AUC sobre datos balanceados con SMOTE."
    ),
    "gradient boosting": (
        "📊 **Gradient Boosting** es uno de los modelos de regresión evaluados. "
        "Combina muchos árboles débiles secuencialmente, donde cada árbol corrige los errores del anterior. "
        "Suele ser el mejor modelo clásico antes de la DNN."
    ),
    "random forest": (
        "🌳 **Random Forest** es el modelo de clasificación ganador. "
        "Crea 100 árboles de decisión en paralelo y combina sus votos. "
        "Es robusto al overfitting y maneja bien el desbalance con SMOTE."
    ),
    "early stopping": (
        "⏹️ **Early Stopping** detiene el entrenamiento de la DNN cuando la pérdida de "
        "validación deja de mejorar por 20 épocas consecutivas. Previene overfitting y "
        "restaura automáticamente los mejores pesos encontrados durante el entrenamiento."
    ),
    "overfitting": (
        "📉 **Overfitting** ocurre cuando el modelo 'memoriza' los datos de entrenamiento "
        "en lugar de generalizar. Para combatirlo usamos:\n\n"
        "- **Dropout** decreciente (0.35 → 0.05)\n"
        "- **BatchNorm**: estabiliza activaciones\n"
        "- **AdamW** con weight_decay (L2 regularization)\n"
        "- **Early Stopping** (patience=20 épocas)\n"
        "- **Gradient Clipping** (max_norm=1.0)"
    ),
    "weighted mse": (
        "⚖️ **WeightedMSE** es la función de pérdida de la DNN. Como solo el 0.17% de las "
        "transacciones son fraude, una MSE normal ignoraría casi todos los fraudes. "
        "WeightedMSE multiplica por 15× el error en transacciones con risk_score alto, "
        "forzando al modelo a aprender mejor los patrones de fraude."
    ),
    "smote": (
        "🔄 **SMOTE** (Synthetic Minority Oversampling Technique) genera ejemplos sintéticos "
        "de la clase minoritaria (fraude) para balancear el entrenamiento. Se aplica SOLO "
        "en el conjunto de entrenamiento para evitar data leakage. Después de SMOTE, "
        "ambas clases tienen el mismo número de muestras."
    ),

    # ── VARIABLES ────────────────────────────────────────────────────────────
    "variables": (
        "🔢 El dataset tiene **31 variables**:\n\n"
        "- **V1–V28**: componentes PCA de datos reales (anonimizados por privacidad)\n"
        "- **Amount**: monto de la transacción en USD/EUR\n"
        "- **Time**: segundos desde la primera transacción del dataset\n"
        "- **Class**: etiqueta objetivo (0 = legítima, 1 = fraude)\n\n"
        "Las más discriminativas: V14, V12, V10, V17, V4, V11 (mayor correlación con fraude)."
    ),
    "pca": (
        "📐 **PCA** (Análisis de Componentes Principales) transforma las variables originales "
        "en componentes no correlacionados que capturan la mayor varianza posible. "
        "Las variables V1–V28 son los 28 componentes principales calculados a partir de datos "
        "reales de transacciones bancarias, **anonimizados** por razones de confidencialidad. "
        "Las más útiles para detectar fraude son V14, V12, V10 y V17."
    ),
    "monto": (
        "💵 El **Amount** es una de las variables más importantes. Los fraudes tienden a tener "
        "montos pequeños (para pasar desapercibidos) o muy grandes (para maximizar el robo). "
        "El sistema lo escala con **RobustScaler** (usa mediana e IQR, resistente a outliers) "
        "antes de pasarlo al modelo."
    ),
    "tiempo": (
        "⏱️ **Time** representa los segundos desde la primera transacción del dataset, no la hora real. "
        "Puede capturar patrones como **ráfagas de transacciones en períodos cortos**, "
        "típico de fraudes automatizados con tarjetas clonadas."
    ),

    # ── ESCALADO ─────────────────────────────────────────────────────────────
    "robust scaler": (
        "📏 **RobustScaler** escala las variables usando la mediana y el IQR (rango intercuartílico) "
        "en lugar de la media y la desviación estándar del StandardScaler. "
        "Esto lo hace más resistente a los outliers extremos que tiene `Amount`. "
        "Se ajusta SOLO en el conjunto de entrenamiento para evitar data leakage."
    ),
    "escalado": (
        "📏 El sistema usa **RobustScaler** para escalar `Amount` y `Time`. "
        "Las variables V1–V28 ya vienen escaladas (son componentes PCA). "
        "El scaler se entrena solo con datos de train y luego se aplica a validación y test "
        "para evitar data leakage (filtración de información del futuro al pasado)."
    ),
    "data leakage": (
        "⚠️ **Data Leakage** ocurre cuando información del conjunto de test 'se filtra' "
        "al entrenamiento, produciendo métricas artificialmente optimistas. "
        "Para evitarlo: el RobustScaler se ajusta SOLO en train, nunca en val/test. "
        "SMOTE también se aplica solo en train."
    ),

    # ── EDA ──────────────────────────────────────────────────────────────────
    "eda": (
        "🔬 El **EDA (Análisis Exploratorio de Datos)** cubre:\n\n"
        "1. Distribución de clases (99.83% legítimas vs 0.17% fraude)\n"
        "2. Distribución de montos por clase\n"
        "3. Correlaciones de variables con `Class` (V14, V12, V10 son las más discriminativas)\n"
        "4. Distribución temporal por clase\n"
        "5. Boxplots de variables PCA críticas\n"
        "6. Estadísticas descriptivas por clase\n\n"
        "También prepara los datasets para clasificación (con SMOTE) y regresión."
    ),
    "desbalance": (
        "⚖️ El dataset tiene **desbalance severo**: 99.83% legítimas, 0.17% fraude (492 de 284,807). "
        "Para el clasificador se usa **SMOTE** y para la DNN se usa **WeightedMSE** "
        "(15× más peso en fraudes). Las métricas prioritarias son F1-score y ROC-AUC, "
        "no la exactitud (accuracy), que sería engañosa en datos tan desbalanceados."
    ),

    # ── MÉTRICAS ─────────────────────────────────────────────────────────────
    "precision": (
        "🎯 **Precisión**: de todos los que el modelo marcó como fraude, ¿cuántos realmente lo eran?\n"
        "Alta precisión = pocos falsos positivos (transacciones legítimas bloqueadas por error).\n\n"
        "**Recall**: de todos los fraudes reales, ¿cuántos detectó el modelo?\n"
        "Alto recall = pocos falsos negativos (fraudes que escaparon sin detección).\n\n"
        "En detección de fraude, el recall suele ser más importante que la precisión."
    ),
    "f1": (
        "📐 El **F1-score** es la media armónica de Precisión y Recall:\n\n"
        "F1 = 2 × (Precision × Recall) / (Precision + Recall)\n\n"
        "Es la métrica principal para seleccionar el mejor clasificador porque "
        "balancea los dos tipos de error. Un modelo con F1 alto funciona bien en ambos."
    ),
    "roc auc": (
        "📈 **ROC-AUC** mide la capacidad del modelo de separar las clases. "
        "Un valor de 1.0 = separación perfecta. Un valor de 0.5 = aleatorio. "
        "Para nuestro dataset desbalanceado, el ROC-AUC es una métrica más fiable "
        "que la exactitud (accuracy)."
    ),
    "r2": (
        "📊 **R²** (coeficiente de determinación) mide qué proporción de la varianza del "
        "risk_score es explicada por el modelo. R²=1 es predicción perfecta, R²=0 equivale "
        "a predecir siempre la media. Para regresión, junto con RMSE y MAE, es la métrica principal."
    ),
    "rmse": (
        "📏 **RMSE** (Root Mean Square Error) es el error promedio del modelo de regresión "
        "en las mismas unidades que el risk_score [0,1]. Penaliza más los errores grandes. "
        "Junto con R² y MAE, determina qué modelo de regresión es el mejor."
    ),
    "metricas": (
        "📊 **Métricas usadas en el proyecto:**\n\n"
        "*Para clasificación:* Precision, Recall, F1-score, ROC-AUC, Matriz de Confusión\n"
        "*Para regresión:* R², RMSE, MAE\n\n"
        "El criterio de selección del mejor clasificador es el **F1-score** y "
        "del mejor regresor es el **R²**. Ambos son robustos ante el desbalance de clases."
    ),

    # ── EXACTITUD ────────────────────────────────────────────────────────────
    "siempre": (
        "❌ El sistema **no siempre acierta**. Comete errores de dos tipos:\n\n"
        "- **Falso Positivo**: marca como fraude una transacción legítima\n"
        "- **Falso Negativo**: no detecta un fraude real\n\n"
        "Por eso se muestra el risk_score continuo en lugar de una decisión binaria, "
        "dejando espacio al analista para decidir."
    ),
    "exactitud": (
        "🎯 Ningún sistema de IA es 100% preciso. La **exactitud global** no es útil aquí "
        "porque con 99.83% de legítimas, un modelo que siempre prediga 'legítima' tendría "
        "99.83% de accuracy pero 0% de recall en fraudes. Por eso usamos F1-score y ROC-AUC."
    ),

    # ── USO DE LA APP ────────────────────────────────────────────────────────
    "ayuda": (
        "❓ **¿Cómo usar la aplicación?**\n\n"
        "1. Ve a la pestaña **🔍 Analizador**\n"
        "2. Ingresa el monto y el tiempo de la transacción\n"
        "3. Expande **Parámetros avanzados** para ajustar V1, V2, V3, V4, V10, V12, V14, V16, V17\n"
        "4. Haz clic en **Analizar transacción**\n"
        "5. Interpreta el semáforo de riesgo, el risk_score y la probabilidad de fraude\n\n"
        "El **Dashboard** (Tab 2) muestra las métricas reales del modelo y el historial de la sesión."
    ),
    "como usar": (
        "🚀 Para analizar una transacción:\n\n"
        "1. Ve a **🔍 Analizador**\n"
        "2. Ingresa el monto en USD y los segundos de tiempo\n"
        "3. Presiona **Analizar transacción**\n"
        "4. Lee el **semáforo** (Bajo/Medio/Alto), el **risk_score** y la **probabilidad de fraude**\n\n"
        "Los resultados se guardan automáticamente en el **Dashboard** (Tab 2)."
    ),
    "dashboard": (
        "📊 El **Dashboard** (Tab 2) muestra:\n\n"
        "- Métricas reales del modelo de regresión (R², RMSE, MAE)\n"
        "- Gráficas generadas por los notebooks: EDA, correlaciones, curvas de aprendizaje\n"
        "- Historial de transacciones analizadas en la sesión actual\n\n"
        "Las métricas y gráficas se cargan automáticamente si los notebooks fueron ejecutados."
    ),

    # ── DATASET ──────────────────────────────────────────────────────────────
    "dataset": (
        "📁 Dataset: **'Credit Card Fraud Detection'** de Kaggle (ULB, Bruselas). "
        "Contiene **284,807 transacciones** de tarjetahabientes europeos (septiembre 2013). "
        "Solo **492 (0.17%)** son fraude. Las variables originales fueron anonimizadas con PCA."
    ),
    "datos": (
        "📊 Los datos provienen de transacciones reales de tarjetas de crédito europeas (2013). "
        "Fueron anonimizados con PCA por razones de confidencialidad. "
        "Solo se conocen `Amount`, `Time` y `Class` en su forma original. "
        "El dataset está disponible en Kaggle de forma libre."
    ),
    "kaggle": (
        "🌐 El dataset se descarga de **Kaggle**: "
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n\n"
        "Una vez descargado, colócalo en la carpeta `data/creditcard.csv` antes de "
        "ejecutar los notebooks."
    ),

    # ── SALUDOS Y CIERRE ─────────────────────────────────────────────────────
    "hola": (
        "👋 ¡Hola! Soy el asistente del **Sistema Antifraude**. "
        "Puedo explicarte cómo funciona el sistema, qué significa cada resultado, "
        "o responder preguntas sobre Machine Learning, fraude y los modelos usados. "
        "¿En qué te ayudo?"
    ),
    "gracias": (
        "😊 ¡Con gusto! Si tienes más preguntas sobre el sistema, los modelos o los resultados, "
        "no dudes en preguntar."
    ),
    "quien eres": (
        "🤖 Soy el **asistente virtual** del Sistema Inteligente de Detección de Fraude "
        "desarrollado para el curso de Introducción a la IA de EAFIT (2026-1). "
        "Puedo explicar cómo funcionan los modelos, qué significan las métricas y "
        "cómo interpretar los resultados del analizador."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# RESPUESTA POR DEFECTO
# ─────────────────────────────────────────────────────────────────────────────

RESPUESTA_DEFAULT = (
    "🤔 No tengo una respuesta específica para eso. Puedes preguntarme sobre:\n\n"
    "- **Niveles de riesgo** (alto / medio / bajo)\n"
    "- **¿Qué es el risk_score?**\n"
    "- **¿Cómo funciona el sistema?**\n"
    "- **¿Qué son las variables V1–V28 y PCA?**\n"
    "- **¿Qué modelos se usan?** (DNN, Random Forest, Gradient Boosting)\n"
    "- **¿Qué es SMOTE o WeightedMSE?**\n"
    "- **¿El sistema siempre acierta?**\n"
    "- **¿Cómo usar la aplicación?**\n"
    "- **¿De dónde vienen los datos?**"
)

# ─────────────────────────────────────────────────────────────────────────────
# SUGERENCIAS DE PREGUNTAS (mostradas al inicio del chat)
# ─────────────────────────────────────────────────────────────────────────────

PREGUNTAS_SUGERIDAS = [
    "¿Qué significa riesgo alto?",
    "¿Cómo funciona el sistema?",
    "¿Qué es el risk score?",
    "¿Qué son las variables V1, V2...?",
    "¿Qué modelos usan?",
    "¿El sistema siempre acierta?",
    "¿Cómo usar la aplicación?",
]

# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE RESPUESTA
# ─────────────────────────────────────────────────────────────────────────────

def _normalizar(texto: str) -> str:
    """Minúsculas + elimina tildes para comparación robusta."""
    texto = texto.lower().strip()
    for c, s in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u'),('ñ','n')]:
        texto = texto.replace(c, s)
    return texto


_STOPWORDS = {
    "que", "como", "cual", "cuales", "de", "del", "la", "el", "los", "las",
    "un", "una", "unos", "unas", "y", "o", "en", "para", "por", "con", "sin",
    "al", "se", "es", "son", "me", "mi", "tu", "su", "sus", "si", "no"
}


def _tokenizar(texto: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_]+", texto))
    normalizados = set()
    for t in tokens:
        if len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        if len(t) >= 3 and t not in _STOPWORDS:
            normalizados.add(t)
    return normalizados


def _responder_local(pregunta: str) -> str:
    """
    Analiza la pregunta y retorna la respuesta más apropiada.
    Ordena las claves de más larga a más corta para que claves específicas
    como 'riesgo alto' no sean capturadas antes por 'riesgo'.
    """
    p = _normalizar(pregunta)
    for clave in sorted(RESPUESTAS.keys(), key=len, reverse=True):
        if _normalizar(clave) in p:
            return RESPUESTAS[clave]

    # Fallback semántico simple por solapamiento de términos clave.
    p_tokens = _tokenizar(p)
    mejor_clave = None
    mejor_score = 0
    mejor_largo = -1

    for clave in RESPUESTAS:
        c_tokens = _tokenizar(_normalizar(clave))
        score = len(p_tokens & c_tokens)
        if score > mejor_score or (score == mejor_score and len(clave) > mejor_largo):
            mejor_clave = clave
            mejor_score = score
            mejor_largo = len(clave)

    if mejor_clave and mejor_score >= 1:
        return RESPUESTAS[mejor_clave]

    return RESPUESTA_DEFAULT


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDER = os.getenv("CHATBOT_PROVIDER", "gemini").strip().lower()
DEFAULT_MODEL = os.getenv("CHATBOT_MODEL", "gemini-1.5-flash")
DEFAULT_GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
).rstrip("/")
API_TIMEOUT_SEC = float(os.getenv("CHATBOT_TIMEOUT_SEC", "20"))


def _leer_metricas_regresion() -> str:
    rutas = [
        BASE_DIR / "reports" / "regression" / "regression_metrics.json",
        BASE_DIR / "reports" / "regression_metrics.json",
    ]
    for ruta in rutas:
        if ruta.exists():
            try:
                data = json.loads(ruta.read_text(encoding="utf-8"))
                mejor = data.get("mejor_modelo", {})
                if mejor:
                    nombre = mejor.get("nombre", "N/D")
                    r2 = mejor.get("r2", "N/D")
                    rmse = mejor.get("rmse", "N/D")
                    mae = mejor.get("mae", "N/D")
                    return (
                        "Metricas de regresion disponibles: "
                        f"mejor_modelo={nombre}, r2={r2}, rmse={rmse}, mae={mae}."
                    )
            except Exception:
                return "No fue posible leer regression_metrics.json."
    return "No hay archivo de metricas de regresion disponible."


def _construir_contexto_proyecto(pregunta: str, top_k: int = 8) -> str:
    consulta = _tokenizar(_normalizar(pregunta))

    documentos = [
        (
            f"Tema: {clave}",
            f"Respuesta interna del proyecto para '{clave}': {texto}",
        )
        for clave, texto in RESPUESTAS.items()
    ]
    documentos.append(
        (
            "Resumen del sistema",
            (
                "Sistema antifraude en Streamlit con 3 modulos: EDA, Clasificacion y Regresion. "
                "Usa variables Time, Amount y V1-V28; clasificacion con Random Forest; "
                "regresion con DNN/Gradient Boosting segun disponibilidad de modelos."
            ),
        )
    )
    documentos.append(("Metricas del proyecto", _leer_metricas_regresion()))

    ranking = []
    for titulo, contenido in documentos:
        tokens_doc = _tokenizar(_normalizar(f"{titulo} {contenido}"))
        score = len(consulta & tokens_doc)
        ranking.append((score, len(tokens_doc), titulo, contenido))

    ranking.sort(key=lambda x: (x[0], x[1]), reverse=True)
    seleccion = [r for r in ranking[:top_k] if r[0] > 0]
    if not seleccion:
        seleccion = ranking[:4]

    bloques = []
    for _, _, titulo, contenido in seleccion:
        bloques.append(f"[{titulo}]\n{contenido}")

    contexto = "\n\n".join(bloques)
    return contexto[:8000]


def _usar_api() -> bool:
    modo = os.getenv("CHATBOT_USE_API", "auto").strip().lower()
    if modo in {"0", "false", "no", "off"}:
        return False
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if modo in {"1", "true", "yes", "on"}:
        return bool(api_key)
    return bool(api_key)


def estado_chatbot() -> str:
    if _usar_api():
        return f"API {DEFAULT_PROVIDER} ({DEFAULT_MODEL}) + contexto del proyecto"
    return "Local (reglas)"


def _responder_api(pregunta: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    contexto = _construir_contexto_proyecto(pregunta)
    system_prompt = (
        "Eres el asistente del Sistema Antifraude del proyecto universitario. "
        "Responde SOLO con la informacion del CONTEXTO_PROYECTO. "
        "Si no encuentras la respuesta en el contexto, di claramente que no tienes ese dato en el proyecto. "
        "No inventes, no cites fuentes externas y responde en espanol de forma breve y clara."
    )
    user_prompt = (
        f"CONTEXTO_PROYECTO:\n{contexto}\n\n"
        f"PREGUNTA_USUARIO:\n{pregunta}\n\n"
        "Da una respuesta util y concreta."
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
        },
    }

    req = request.Request(
        url=f"{DEFAULT_GEMINI_BASE_URL}/models/{DEFAULT_MODEL}:generateContent?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=API_TIMEOUT_SEC) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parts = body["candidates"][0]["content"]["parts"]
        content = "".join(part.get("text", "") for part in parts).strip()
        return content or None
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError):
        return None


def responder(pregunta: str) -> str:
    if _usar_api():
        respuesta_api = _responder_api(pregunta)
        if respuesta_api:
            return respuesta_api
    return _responder_local(pregunta)


# ─────────────────────────────────────────────────────────────────────────────
# TEST BÁSICO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "¿Qué significa riesgo alto?",
        "¿Qué es el risk score?",
        "¿Cómo funciona la red neuronal?",
        "¿Qué es SMOTE?",
        "¿Qué es WeightedMSE?",
        "¿Qué es early stopping?",
        "¿Qué son V1 y V2?",
        "¿Qué es PCA?",
        "¿El monto influye?",
        "¿Siempre acierta?",
        "¿Qué métricas usan?",
        "¿Qué es el F1?",
        "¿Qué es data leakage?",
        "hola",
        "¿Quién eres?",
        "xyz sin respuesta",
    ]
    for p in tests:
        r = responder(p)
        print(f"Q: {p}\nA: {r[:120]}...\n{'-'*60}")
