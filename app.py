import streamlit as st
import pandas as pd
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Clave de NewsAPI desde secretos de Streamlit
API_KEY = st.secrets.get("NEWSAPI_KEY")

# Carga del modelo FinBERT
@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_classifier()
plt.rcParams.update({"figure.autolayout": True})

# Función para obtener noticias via NewsAPI
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
ticker = st.text_input("Ingrese el ticker (ej: AAPL, GOOGL, QQQ)", value="QQQ").upper()
if st.button("Obtener noticias"):
    with st.spinner("Descargando noticias via NewsAPI..."):
        df = get_news(ticker)

    # Verificar que df no esté vacío
    if df.empty:
        st.warning("No se encontraron noticias para el ticker proporcionado o respuesta vacía.")
    else:
        # Mostrar contenido crudo de las noticias descargadas
        st.subheader("Datos crudos de noticias descargadas")
        st.dataframe(df)

        # Renombrar y convertir fechas
        df = df.rename(columns={"publishedAt": "pubDate", "description": "summary"})
        df["pubDate"] = pd.to_datetime(df["pubDate"], utc=True)

        # Análisis de sentimiento
        sentiments = df["summary"].apply(lambda x: classifier(x or "")[0])
        df["sentiment"] = sentiments.apply(lambda r: r["label"].lower())
        df["confidence"] = sentiments.apply(lambda r: r["score"])

        # Mostrar tabla de resultados
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
