import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datetime import datetime
import time
import os

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(page_title="Contador de Personas - Carnaval Nariño", layout="wide")
st.title("🎭 Contador de Personas con Densidad en Tiempo Real")
st.markdown("Apunta la cámara a la multitud y obtén densidad en personas/m²")

# ============================================
# SIDEBAR: CONFIGURACIÓN
# ============================================
with st.sidebar:
    st.header("Configuración")
    
    area_visible = st.number_input("Área visible de la cámara (m²)", min_value=1.0, value=30.0, step=5.0,
                                   help="Mide aproximadamente el área que cubre la cámara (ej: a 5m de altura ≈ 30-50 m²)")
    
    conf_threshold = st.slider("Umbral de confianza YOLO", 0.1, 1.0, 0.4, 0.05)
    
    st.markdown("---")
    st.caption("Modelo: YOLOv8s (preentrenado en personas)")
    st.caption("Presiona 'Iniciar Cámara' para comenzar")

# ============================================
# CARGAR MODELO
# ============================================
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")  # Cambia a yolov8m.pt o yolov8l.pt para más precisión

model = load_model()

# ============================================
# ESTADO DE LA APP
# ============================================
if "running" not in st.session_state:
    st.session_state.running = False
if "data" not in st.session_state:
    st.session_state.data = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# ============================================
# CONTROLES
# ============================================
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Iniciar Cámara", type="primary"):
        st.session_state.running = True
        st.session_state.data = []
        st.rerun()

with col2:
    if st.button("Detener y Generar Reporte"):
        st.session_state.running = False
        st.rerun()

with col3:
    if len(st.session_state.data) > 0:
        if st.button("Limpiar Datos"):
            st.session_state.data = []
            st.rerun()

# ============================================
# FRAME PLACEHOLDER
# ============================================
frame_placeholder = st.empty()
info_placeholder = st.empty()
chart_placeholder = st.empty()

# ============================================
# CAPTURA DE CÁMARA
# ============================================
cap = cv2.VideoCapture(0)

if st.session_state.running:
    st.session_state.start_time = time.time()

while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.error("No se pudo acceder a la cámara")
        break

    # Inferencia YOLO
    results = model(frame, conf=conf_threshold, classes=[0])[0]
    personas = len(results.boxes) if results.boxes is not None else 0
    densidad = personas / area_visible if area_visible > 0 else 0

    # Dibujar bounding boxes
    annotated_frame = results.plot()  # YOLO tiene método plot() que dibuja bonito

    # Textos en el frame
    cv2.putText(annotated_frame, f"Personas: {personas}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    cv2.putText(annotated_frame, f"Densidad: {densidad:.2f} pers/m²", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)

    # Guardar datos
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.data.append({
        "timestamp": ts,
        "personas": personas,
        "densidad_pers_m2": round(densidad, 3)
    })

    # Mostrar frame en Streamlit
    frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

    # Info en tiempo real
    clasificacion = "BAJA" if densidad < 1 else "MEDIA" if densidad < 2 else "ALTA" if densidad < 3 else "¡MUY ALTA!"
    info_placeholder.markdown(f"""
    **Estado actual**  
    Personas detectadas: **{personas}**  
    Densidad: **{densidad:.2f} personas/m²** → **{clasificacion}**  
    Registros capturados: {len(st.session_state.data)}
    """)

    # Gráfico en vivo
    if len(st.session_state.data) > 10:
        df_live = pd.DataFrame(st.session_state.data)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_live["densidad_pers_m2"], color="red", linewidth=2)
        ax.set_title("Densidad en tiempo real (personas/m²)")
        ax.set_ylabel("Densidad")
        ax.grid(True, alpha=0.3)
        chart_placeholder.pyplot(fig)
        plt.close(fig)

    time.sleep(0.03)  # ~30 FPS
    st.rerun()

# Liberar cámara al detener
cap.release()

# ============================================
# REPORTE FINAL Y DESCARGA CSV
# ============================================
if len(st.session_state.data) > 0 and not st.session_state.running:
    df = pd.DataFrame(st.session_state.data)
    
    st.success(f"¡Captura finalizada! {len(df)} registros guardados.")
    
    # Estadísticas
    dens_prom = df["densidad_pers_m2"].mean()
    dens_max = df["densidad_pers_m2"].max()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Densidad promedio", f"{dens_prom:.2f} pers/m²")
    col2.metric("Densidad máxima", f"{dens_max:.2f} pers/m²")
    col3.metric("Personas promedio visibles", f"{df['personas'].mean():.1f}")
    
    # Gráfico final
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["personas"], label="Personas detectadas", color="blue")
    ax.plot(df["densidad_pers_m2"] * 20, label="Densidad x20 (escala)", color="red", alpha=0.7)
    ax.set_title("Evolución durante la captura")
    ax.set_xlabel("Tiempo (frames)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    # Descarga CSV
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="📥 Descargar CSV con todos los datos",
        data=csv,
        file_name=f"conteo_personas_densidad_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

else:
    st.info("Presiona 'Iniciar Cámara' para comenzar el conteo en vivo.")
