import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image, ImageOps  # Importamos ImageOps
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Pepe IA: Contador de Racimos", layout="wide")
st.title("🍇 Pepe IA: Conteo Automático de Racimos")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración del Modelo")
conf_val = st.sidebar.select_slider(
    "Umbral de Confianza",
    options=[round(i * 0.05, 2) for i in range(1, 20)],
    value=0.25
)

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    model_path = os.path.join('runs', 'detect', 'racimos_yolo26_v2', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        st.error(f"❌ No se encontró el modelo en: {model_path}")
        st.stop()
    return YOLO(model_path)

model = load_model()

# --- INTERFAZ DE SUBIDA ---
uploaded_file = st.file_uploader("Sube la imagen del viñedo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Abrir la imagen
    img_raw = Image.open(uploaded_file)
    
    # 2. CORRECCIÓN DE ROTACIÓN (EXIF Transpose)
    # Esto soluciona el problema de las imágenes que salen giradas
    img = ImageOps.exif_transpose(img_raw)
    
    with st.spinner("Pepe está analizando la imagen..."):
        # EJECUTAR PREDICCIÓN
        results = model.predict(source=img, conf=conf_val, imgsz=1024, device='cuda:0')
        result = results[0]

        # --- MOSTRAR COMPARATIVA ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen Original")
            st.image(img, use_container_width=True)
        
        with col2:
            num_racimos = len(result.boxes)
            st.subheader(f"Predicción ({num_racimos} racimos)")
            
            # Dibujar cajas sin etiquetas ni confianza
            res_plotted = result.plot(labels=False, conf=False)
            st.image(res_plotted, channels="BGR", use_container_width=True)

        # --- SECCIÓN DE GUARDADO ---
        st.write("---")
        if st.button("💾 Guardar Resultado"):
            output_dir = "outputs_directos"
            os.makedirs(output_dir, exist_ok=True)
            
            name_only = os.path.splitext(uploaded_file.name)[0]
            save_path = os.path.join(output_dir, f"det_{name_only}.jpg")
            
            # Guardar la imagen procesada
            Image.fromarray(res_plotted[:,:,::-1]).save(save_path)
            st.success(f"✅ Imagen guardada en: {save_path}")

        st.info(f"Se han detectado {num_racimos} objetos con confianza > {conf_val}")