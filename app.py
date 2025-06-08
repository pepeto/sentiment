import streamlit as st
import pandas as pd
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Clave de NewsAPI: reemplaza aquí o, idealmente, configúralo en Streamlit Secrets
API_KEY = st.secrets.get("NEWSAPI_KEY")

# Carga del modelo FinBERT (se cachea para no recargar en cada interacción)
@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_classifier()
plt.rcParams.update({"figure.autolayout": True})

# Función para obtener noticias desde NewsAPI
@st.cache_data
def get_news(ticker: str) -> pd.DataFrame:
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        "sortBy=publishedAt&"
        "language=en&"
        f"apiKey={API_KEY}"
    )
    resp = requests.get(url)
    articles = resp.json().get("articles", [])
    return pd.DataFrame(articles).head(10)

# Interfaz Streamlit
st.title("Análisis de Sentimiento de Noticias Financieras")

ticker = st.text_input("Ingrese el ticker (ej: AAPL, GOOGL, QQQ)", value="QQQ").upper()

if st.button("Obtener noticias"):
    with st.spinner("Descargando noticias via NewsAPI..."):
        df = get_news(ticker)

    if df.empty:
        st.warning("No se encontraron noticias para el ticker proporcionado.")
    else:
        # Renombrar/extraer campos
        df = df.rename(columns={"publishedAt": "pubDate", "description": "summary"})
        df["pubDate"] = pd.to_datetime(df["pubDate"], utc=True)

        # Análisis de sentimiento sobre el resumen
        sentiments = df["summary"].apply(lambda x: classifier(x or "")[0])
        df["sentiment"] = sentiments.apply(lambda r: r["label"].lower())
        df["confidence"] = sentiments.apply(lambda r: r["score"])

        # Mostrar resultados
        st.subheader("Resultados de las primeras 10 noticias")
        st.dataframe(df[["title", "pubDate", "sentiment", "confidence"]])

        # Gráfico de distribución
        st.subheader("Distribución de Sentimientos")
        fig, ax = plt.subplots()
        df["sentiment"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Sentimiento")
        ax.set_ylabel("Cantidad")
        ax.set_title(f"Sentimientos para {ticker}")
        st.pyplot(fig)
