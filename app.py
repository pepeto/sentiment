import streamlit as st
import pandas as pd
import yfinance as yf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Carga del modelo FinBERT (se cachea para no recargar en cada interacción)
@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_classifier()
plt.rcParams.update({"figure.autolayout": True})

# Título de la app
st.title("Análisis de Sentimiento de Noticias Financieras")

# Input de ticker
ticker = st.text_input("Ingrese el ticker (ej: AAPL, GOOGL, QQQ)", value="QQQ").upper()

if st.button("Obtener noticias"):
    with st.spinner("Descargando y procesando noticias..."):
        data = yf.Ticker(ticker)
        # Obtener noticias y chequear que no esté vacío
        news_items = getattr(data, 'news', []) or []

    # Verificar contenido de news_items
    if not news_items:
        st.warning("No se encontraron noticias para el ticker proporcionado o la respuesta está vacía.")
        st.warning("No se encontraron noticias para el ticker proporcionado.")
    else:
        # DataFrame de las primeras 10 noticias
        df = pd.DataFrame(news_items).head(10)

        # Mostrar muestra cruda de los datos descargados
        st.subheader("Muestra de datos crudos descargados")
        st.write(df)

        # Convertir pubDate (ISO8601) a datetime
        df["pubDate"] = pd.to_datetime(df["pubDate"], utc=True)

        # Análisis de sentimiento basado en el resumen
        sentiments = df["summary"].apply(lambda x: classifier(x)[0])
        df["sentiment"] = sentiments.apply(lambda res: res["label"].lower())
        df["confidence"] = sentiments.apply(lambda res: res["score"])

        # Mostrar tabla de resultados
        st.subheader("Resultados de las primeras 10 noticias")
        st.dataframe(df[["title", "pubDate", "sentiment", "confidence"]])

        # Gráfico de distribución de sentimientos
        st.subheader("Distribución de Sentimientos")
        fig, ax = plt.subplots()
        df["sentiment"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Sentimiento")
        ax.set_ylabel("Cantidad")
        ax.set_title(f"Sentimientos para {ticker}")
        st.pyplot(fig)
