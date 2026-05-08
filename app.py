import streamlit as st
import pickle
from proyecto import (OPCIONES_INTERES, OPCIONES_NIVEL)

st.set_page_config(page_title="Recomendador de rutas", layout="centered")
st.title("Recomendador de rutas de aprendizaje")
st.markdown("Este modelo de inteligencia artificial te sugiere una ruta de aprendizaje a partir de tu interés, tu nivel y tu experiencia.")
st.markdown("Responde con el número de la opción que mejor describa tu perfil.")

# se lee el modelo
with open("modelo.pkl", "rb") as f:
    modelo, le_area, columnas_caracteristicas = pickle.load(f)

interes = st.selectbox("Área de interés", list(OPCIONES_INTERES.values()))
nivel = st.selectbox("Nivel actual", list(OPCIONES_NIVEL.values()))
experiencia = st.number_input("Años de experiencia", min_value=0.0, step=0.5)