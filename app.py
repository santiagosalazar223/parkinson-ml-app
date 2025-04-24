import streamlit as st
import pandas as pd
import joblib

# --- Cargar modelo y columnas ---
clf = joblib.load("modelo_parkinson.pkl")
columnas = joblib.load("columnas_parkinson.pkl")  # Asegúrate que tenga solo columnas limpias y sin duplicados

# --- Título de la app ---
st.title("🧠 Predicción de Parkinson a partir de voz")
st.markdown("Usa características acústicas para predecir si una persona podría tener Parkinson.")

# --- Inputs del usuario ---
jitter = st.slider("Jitter (%)", 0.0, 0.02, 0.005)
shimmer = st.slider("Shimmer", 0.0, 0.05, 0.02)
hnr = st.slider("HNR (Harmonic-to-Noise Ratio)", 5.0, 40.0, 15.0)

# --- Crear el input con todas las columnas en 0 ---
input_data = pd.DataFrame([[0] * len(columnas)], columns=columnas)

# --- Insertar los valores reales en las columnas correctas ---
# Asegúrate que estos nombres coincidan EXACTAMENTE con los de las columnas originales
input_data["MDVP:Jitter(%)"] = jitter
input_data["MDVP:Shimmer"] = shimmer
input_data["HNR"] = hnr

# --- Asegurar el orden correcto ---
input_data = input_data[columnas]

# --- Hacer predicción ---
pred = clf.predict(input_data)[0]
proba = clf.predict_proba(input_data)[0][1]  # probabilidad de Parkinson

# --- Mostrar resultado ---
if pred == 1:
    st.error(f"🧠 El modelo predice: Parkinson (probabilidad: {proba:.2%})")
else:
    st.success(f"✅ El modelo predice: No Parkinson (probabilidad: {proba:.2%})")
