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
        news_items = data.news

    if not news_items:
        st.warning("No se encontraron noticias para el ticker proporcionado.")
    else:
        # DataFrame de las primeras 10 noticias
        df = pd.DataFrame(news_items).head(10)
        
        # Determinar la columna de fecha
        if "providerPublishTime" in df.columns:
            time_col = "providerPublishTime"
        elif "provider_publish_time" in df.columns:
            time_col = "provider_publish_time"
        else:
            time_col = None

        if time_col:
            df["pubDate"] = pd.to_datetime(df[time_col], unit="s")
        else:
            df["pubDate"] = pd.NaT
            st.warning("No se encontró campo de fecha en los datos de noticias.")

        # Análisis de sentimiento y nivel de confianza
        df["sentiment"] = df["summary"].apply(lambda x: classifier(x)[0]["label"].lower())
        df["confidence"] = df["summary"].apply(lambda x: classifier(x)[0]["score"])

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
