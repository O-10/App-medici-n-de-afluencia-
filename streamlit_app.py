import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datetime import datetime
import time

st.set_page_config(page_title="Contador Personas Carnaval Nariño", layout="wide")
st.title("🎭 Contador de Personas con Densidad en Tiempo Real")
st.markdown("Apunta la cámara a la multitud y obtén densidad en personas/m²")

with st.sidebar:
    st.header("Configuración")
    area_visible = st.number_input("Área visible de la cámara (m²)", min_value=1.0, value=30.0, step=5.0)
    conf_threshold = st.slider("Umbral de confianza YOLO", 0.1, 1.0, 0.4, 0.05)
    st.markdown("---")
    st.caption("Modelo: YOLOv8s (preentrenado en personas)")

@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

if "running" not in st.session_state:
    st.session_state.running = False
if "data" not in st.session_state:
    st.session_state.data = []

col1, col2 = st.columns(2)
with col1:
    if st.button("Iniciar Cámara", type="primary"):
        st.session_state.running = True
        st.session_state.data = []
        st.rerun()
with col2:
    if st.button("Detener y Generar Reporte"):
        st.session_state.running = False
        st.rerun()

frame_placeholder = st.empty()
info_placeholder = st.empty()
chart_placeholder = st.empty()

# Intentar acceder a la cámara
cap = cv2.VideoCapture(0)

if st.session_state.running:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo acceder a la cámara. Prueba en modo local o permite acceso.")
            break

        results = model(frame, conf=conf_threshold, classes=[0])[0]
        personas = len(results.boxes) if results.boxes is not None else 0
        densidad = personas / area_visible if area_visible > 0 else 0

        annotated_frame = results.plot()

        cv2.putText(annotated_frame, f"Personas: {personas}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.putText(annotated_frame, f"Densidad: {densidad:.2f} pers/m²", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)

        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.data.append({
            "timestamp": ts,
            "personas": personas,
            "densidad_pers_m2": round(densidad, 3)
        })

        frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        clasificacion = "BAJA" if densidad < 1 else "MEDIA" if densidad < 2 else "ALTA" if densidad < 3 else "¡MUY ALTA!"
        info_placeholder.markdown(f"""
        **Estado actual**  
        Personas detectadas: **{personas}**  
        Densidad: **{densidad:.2f} personas/m²** → **{clasificacion}**  
        Registros: {len(st.session_state.data)}
        """)

        if len(st.session_state.data) > 10:
            df_live = pd.DataFrame(st.session_state.data)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_live["densidad_pers_m2"], color="red", linewidth=2)
            ax.set_title("Densidad en tiempo real")
            ax.grid(True, alpha=0.3)
            chart_placeholder.pyplot(fig)
            plt.close(fig)

        time.sleep(0.03)
        st.rerun()

cap.release()

if len(st.session_state.data) > 0 and not st.session_state.running:
    df = pd.DataFrame(st.session_state.data)
    st.success(f"¡Captura finalizada! {len(df)} registros")

    col1, col2, col3 = st.columns(3)
    col1.metric("Densidad promedio", f"{df['densidad_pers_m2'].mean():.2f} pers/m²")
    col2.metric("Densidad máxima", f"{df['densidad_pers_m2'].max():.2f} pers/m²")
    col3.metric("Personas promedio", f"{df['personas'].mean():.1f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["personas"], label="Personas", color="blue")
    ax.plot(df["densidad_pers_m2"] * 20, label="Densidad x20", color="red")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="📥 Descargar CSV completo",
        data=csv,
        file_name=f"conteo_carnaval_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

else:
    st.info("Presiona 'Iniciar Cámara' para comenzar el conteo en vivo.")
