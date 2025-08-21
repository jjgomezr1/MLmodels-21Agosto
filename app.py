import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from faker import Faker
import random

# --------------------------
# Función para generar datos sintéticos
# --------------------------
def generar_datos(n_muestras, n_columnas):
    fake = Faker()
    deportes = ["Fútbol", "Baloncesto", "Tenis", "Atletismo", "Natación", "Ciclismo"]
    posiciones = ["Delantero", "Defensa", "Portero", "Base", "Escolta", "Nadador", "Corredor"]

    data = {}

    # Variables posibles
    posibles_columnas = {
        "Deporte": lambda: random.choice(deportes),
        "Jugador": lambda: fake.first_name(),
        "Posición": lambda: random.choice(posiciones),
        "Edad": lambda: np.random.randint(15, 40),
        "Altura_cm": lambda: np.random.randint(150, 210),
        "Peso_kg": lambda: np.random.randint(50, 100),
        "Puntos": lambda: np.random.randint(0, 30),
        "Asistencias": lambda: np.random.randint(0, 10),
        "Victorias": lambda: np.random.choice([0, 1]),
    }

    # Selección aleatoria de columnas
    columnas = list(posibles_columnas.keys())[:n_columnas]

    for col in columnas:
        data[col] = [posibles_columnas[col]() for _ in range(n_muestras)]

    return pd.DataFrame(data)

# --------------------------
# Interfaz con Streamlit
# --------------------------
st.set_page_config(page_title="EDA Deportivo", layout="wide")
st.title("⚽🏀 Análisis Exploratorio de Datos (EDA) en Deportes")

# Parámetros de entrada
st.sidebar.header("Configuración de datos")
n_muestras = st.sidebar.slider("Número de muestras", min_value=50, max_value=500, value=200, step=50)
n_columnas = st.sidebar.slider("Número de columnas", min_value=3, max_value=6, value=5)

# Generar dataset
df = generar_datos(n_muestras, n_columnas)

st.subheader("📊 Dataset generado")
st.dataframe(df)

# Selección de columnas
st.sidebar.header("Exploración de datos")
columnas = st.sidebar.multiselect("Selecciona columnas para analizar", df.columns, default=df.columns.tolist())

# Selección de tipo de gráfico
graficos = ["Histograma", "Gráfico de barras", "Gráfico de dispersión", "Gráfico de pastel", "Tendencia temporal"]
grafico = st.sidebar.radio("Selecciona tipo de gráfico", graficos)

# --------------------------
# Visualizaciones
# --------------------------
st.subheader("📈 Visualización de datos")

if grafico == "Histograma":
    col = st.selectbox("Selecciona columna numérica", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

elif grafico == "Gráfico de barras":
    col = st.selectbox("Selecciona columna categórica", df.select_dtypes(exclude=np.number).columns)
    fig, ax = plt.subplots()
    df[col].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

elif grafico == "Gráfico de dispersión":
    col_x = st.selectbox("Eje X", df.select_dtypes(include=np.number).columns, index=0)
    col_y = st.selectbox("Eje Y", df.select_dtypes(include=np.number).columns, index=1)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax)
    st.pyplot(fig)

elif grafico == "Gráfico de pastel":
    col = st.selectbox("Selecciona columna categórica", df.select_dtypes(exclude=np.number).columns)
    fig, ax = plt.subplots()
    df[col].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

elif grafico == "Tendencia temporal":
    if "Puntos" in df.columns:
        df_temp = df["Puntos"].cumsum()
        fig, ax = plt.subplots()
        df_temp.plot(ax=ax, title="Tendencia acumulada de puntos")
        st.pyplot(fig)
    else:
        st.warning("No hay columna 'Puntos' para graficar tendencia")

# --------------------------
# Estadísticas descriptivas
# --------------------------
st.subheader("📑 Estadísticas descriptivas")
st.write(df.describe(include="all"))
