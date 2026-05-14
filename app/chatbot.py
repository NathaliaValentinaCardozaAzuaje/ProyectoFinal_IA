"""
chatbot.py — Bot conversacional para el Sistema Antifraude
Usa keyword matching con normalización de tildes.
"""

RESPUESTAS = {
    "riesgo alto": (
        "🚨 **Riesgo Alto** (score > 0.6): el modelo detectó patrones muy similares a fraudes históricos. "
        "No garantiza fraude, pero se recomienda verificación adicional antes de aprobar."
    ),
    "riesgo medio": (
        "⚠️ **Riesgo Medio** (score 0.3–0.6): la transacción tiene algunas características inusuales "
        "pero no suficientes para catalogarla como fraude. Se sugiere monitoreo adicional."
    ),
    "riesgo bajo": (
        "✅ **Riesgo Bajo** (score < 0.3): la transacción tiene características similares a operaciones "
        "legítimas. El sistema la considera segura, aunque ningún modelo es 100% infalible."
    ),
    "fraude": (
        "🔍 El sistema detecta fraude analizando patrones en múltiples variables: el monto, el tiempo "
        "y 28 variables PCA del comportamiento de la tarjeta (V1–V28). Cuando esos patrones coinciden "
        "con fraudes históricos del dataset de 284,807 transacciones reales, se emite una alerta."
    ),
    "risk score": (
        "📊 El **risk_score** es un valor continuo entre 0.0 y 1.0:\n\n"
        "- 🟢 0.0–0.3 → Bajo riesgo\n"
        "- 🟡 0.3–0.6 → Riesgo medio\n"
        "- 🔴 0.6–1.0 → Alto riesgo\n\n"
        "Se calcula combinando: fraude confirmado (60%), monto normalizado (30%) y magnitud PCA (10%)."
    ),
    "puntaje": (
        "📊 El **puntaje de riesgo** es un número entre 0 y 1. Cuanto más cercano a 1, "
        "mayor la probabilidad de que la transacción sea fraudulenta. "
        "Es producido por el modelo de regresión (Gradient Boosting)."
    ),
    "que hace": (
        "🤖 Este sistema analiza transacciones financieras con Machine Learning:\n\n"
        "1. **Modelo de regresión** → predice un puntaje continuo de riesgo\n"
        "2. **Modelo de clasificación** → decide si la transacción es fraude o no\n\n"
        "Entrenado con 284,807 transacciones reales del dataset Kaggle Credit Card Fraud."
    ),
    "sistema": (
        "🏦 El **Sistema Antifraude** evalúa transacciones bancarias en tiempo real con IA. "
        "Ingresa los datos de una transacción y obtén en segundos una estimación del riesgo. "
        "Usa la pestaña **🔍 Analizador** para probarlo."
    ),
    "modelo": (
        "⚙️ Usamos **dos modelos de Machine Learning**:\n\n"
        "- **Regresión (Gradient Boosting):** predice el puntaje continuo de riesgo (0–1)\n"
        "- **Clasificación (Random Forest):** decide si es fraude (1) o no (0)\n\n"
        "Ambos entrenados con el dataset 'Credit Card Fraud Detection' de Kaggle (ULB)."
    ),
    "regresion": (
        "📈 El **modelo de regresión** predice un puntaje continuo de riesgo. "
        "Se compararon: Linear Regression, Random Forest y Gradient Boosting. "
        "El mejor fue seleccionado con base en el R² más alto y el RMSE más bajo."
    ),
    "clasificacion": (
        "🎯 El **modelo de clasificación** predice si la transacción es fraude (1) o legítima (0). "
        "Fue construido con técnicas de balanceo SMOTE para manejar el desbalance extremo de clases."
    ),
    "variables": (
        "🔢 El dataset tiene **31 variables**:\n\n"
        "- **V1–V28**: transformación PCA de datos reales (anonimizados)\n"
        "- **Amount**: monto de la transacción en USD\n"
        "- **Time**: segundos desde la primera transacción del dataset\n"
        "- **Class**: etiqueta (0 = legítima, 1 = fraude)"
    ),
    "pca": (
        "📐 **PCA** transforma las variables originales en componentes no correlacionados. "
        "Las variables V1–V28 son los 28 componentes principales calculados a partir de datos "
        "reales de transacciones bancarias, anonimizados por razones de confidencialidad."
    ),
    "monto": (
        "💵 El **monto** es una de las variables más importantes. Los fraudes tienden a tener "
        "montos pequeños (para pasar desapercibidos) o muy grandes (para maximizar el robo). "
        "El sistema lo normaliza automáticamente antes de predecir."
    ),
    "tiempo": (
        "⏱️ **Time** representa los segundos desde la primera transacción del dataset, no la hora real. "
        "Puede capturar patrones como transacciones agrupadas en períodos cortos, "
        "típico de fraudes automatizados."
    ),
    "precis": (
        "🎯 Ningún sistema de IA es 100% preciso. Puede haber:\n\n"
        "- **Falsos positivos**: transacciones legítimas marcadas como fraude\n"
        "- **Falsos negativos**: fraudes que no son detectados\n\n"
        "El modelo fue evaluado con R², RMSE, MAE, Precision, Recall y ROC-AUC."
    ),
    "siempre": (
        "❌ El sistema **no siempre acierta**. Comete errores de dos tipos:\n\n"
        "- **Falso Positivo**: marca como fraude una transacción legítima\n"
        "- **Falso Negativo**: no detecta un fraude real\n\n"
        "Por eso se muestra un puntaje continuo en lugar de una decisión binaria."
    ),
    "ayuda": (
        "❓ **¿Cómo usar la aplicación?**\n\n"
        "1. Ve a la pestaña **🔍 Analizador**\n"
        "2. Ingresa el monto y el tiempo de la transacción\n"
        "3. Expande los parámetros avanzados para ajustar V1–V17\n"
        "4. Haz clic en **Analizar transacción**\n"
        "5. Interpreta el semáforo de riesgo y el puntaje obtenido"
    ),
    "como usar": (
        "🚀 Para analizar una transacción:\n\n"
        "1. Ve a **🔍 Analizador**\n"
        "2. Ingresa el monto en USD y el tiempo\n"
        "3. Presiona **Analizar transacción**\n"
        "4. El sistema mostrará el puntaje de riesgo y la probabilidad de fraude"
    ),
    "dataset": (
        "📁 Dataset: **'Credit Card Fraud Detection'** de Kaggle (ULB, Bruselas). "
        "Contiene **284,807 transacciones** de tarjetahabientes europeos (septiembre 2013), "
        "de las cuales solo **492 (0.17%)** son fraude."
    ),
    "datos": (
        "📊 Los datos provienen de transacciones reales de tarjetas de crédito europeas (2013). "
        "Fueron anonimizados con PCA. El dataset está altamente desbalanceado: "
        "solo el 0.17% de las transacciones son fraude."
    ),
    "desbalance": (
        "⚖️ El dataset tiene **desbalance severo**: 99.83% legítimas, 0.17% fraude. "
        "Para el clasificador se usó **SMOTE** (Synthetic Minority Oversampling) "
        "para generar ejemplos sintéticos de fraude y balancear el entrenamiento."
    ),
    "hola": (
        "👋 ¡Hola! Soy el asistente del **Sistema Antifraude**. "
        "Puedo explicarte cómo funciona el sistema, qué significa cada resultado "
        "y responder preguntas sobre fraude y Machine Learning. ¿En qué te ayudo?"
    ),
    "gracias": (
        "😊 ¡Con gusto! Si tienes más preguntas sobre el sistema o los resultados, "
        "no dudes en preguntar."
    ),
}

RESPUESTA_DEFAULT = (
    "🤔 No tengo una respuesta específica para eso. Puedes preguntarme sobre:\n\n"
    "- **Riesgo alto / medio / bajo**\n"
    "- **¿Cómo funciona el sistema?**\n"
    "- **¿Qué es el risk score?**\n"
    "- **¿Qué son las variables V1–V28 y PCA?**\n"
    "- **¿El sistema siempre acierta?**\n"
    "- **¿Cómo usar la aplicación?**\n"
    "- **¿De dónde vienen los datos?**"
)

PREGUNTAS_SUGERIDAS = [
    "¿Qué significa riesgo alto?",
    "¿Cómo funciona el sistema?",
    "¿Qué son las variables V1, V2...?",
    "¿El sistema siempre acierta?",
    "¿Cómo usar la aplicación?",
    "¿Qué es el risk score?",
    "¿De dónde vienen los datos?",
]


def _normalizar(texto: str) -> str:
    """Minúsculas + elimina tildes para comparación robusta."""
    texto = texto.lower().strip()
    for c, s in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ü','u'),('ñ','n')]:
        texto = texto.replace(c, s)
    return texto


def responder(pregunta: str) -> str:
    """
    Analiza la pregunta y retorna la respuesta más apropiada.
    Ordena las claves de más específica (larga) a más general para evitar
    que 'riesgo' capture antes que 'riesgo alto'.
    """
    p = _normalizar(pregunta)
    for clave in sorted(RESPUESTAS.keys(), key=len, reverse=True):
        if _normalizar(clave) in p:
            return RESPUESTAS[clave]
    return RESPUESTA_DEFAULT


if __name__ == "__main__":
    tests = [
        "¿Qué significa riesgo alto?",
        "¿Por qué fue fraude?",
        "¿Qué hace el sistema?",
        "¿Qué son V1 y V2?",
        "¿Qué es PCA?",
        "¿El monto influye?",
        "¿Siempre acierta?",
        "hola",
        "xyz sin respuesta",
    ]
    for p in tests:
        print(f"Q: {p}\nA: {responder(p)[:100]}...\n")
