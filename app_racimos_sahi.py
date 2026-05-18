import streamlit as st
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
from PIL import Image, ImageOps
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Pepe IA: Contador de Racimos", layout="wide")
st.title("🍇 Pepe IA: Contador de Racimos con Slicing")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración de IA")
conf_val = st.sidebar.select_slider(
    "Umbral de Confianza (Confidence)",
    options=[round(i * 0.05, 2) for i in range(1, 20)],
    value=0.25,
    help="Menor confianza = más detecciones (posibles falsos positivos). Mayor confianza = más exigente."
)

# Carpeta donde guardas las fotos etiquetadas manualmente para comparar
VIS_BASE_DIR = "visualizacion_dataset_racimos"

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model(_conf):
    # Ruta al modelo de racimos
    model_path = os.path.join('runs', 'detect', 'racimos_sliced_yolo26_v1', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        st.error(f"❌ No se encontró el modelo en: {model_path}")
        st.stop()
        
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=_conf,
        device='cuda:0', # Cambiar a 'cpu' si no hay GPU disponible
    )

model = load_model(conf_val)

# --- INTERFAZ DE SUBIDA ---
uploaded_file = st.file_uploader("Sube la imagen del viñedo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Abrir imagen original
    img_raw = Image.open(uploaded_file)
    
    # 2. Corregir rotación automática (EXIF)
    img = ImageOps.exif_transpose(img_raw)
    
    # Convertir a formato compatible con SAHI (Numpy Array)
    img_array = np.array(img)

    with st.spinner("Pepe está analizando la imagen por trozos (Slicing)..."):
        # 3. Ejecutar predicción con SAHI
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=800,
            slice_width=800,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        # 4. Filtrar detecciones (solo queremos racimos)
        result.object_prediction_list = [
            pred for pred in result.object_prediction_list if "racimo" in pred.category.name.lower()
        ]
        
        num_racimos = len(result.object_prediction_list)

        # --- MOSTRAR COMPARATIVA ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen Original")
            st.image(img, use_container_width=True)
        
        with col2:
            st.subheader(f"🤖 Predicción de Pepe ({num_racimos} racimos)")
            
            # Crear visualización temporal para la web
            temp_web_dir = "temp_web_racimos"
            os.makedirs(temp_web_dir, exist_ok=True)
            
            # Exportamos visuales ocultando etiquetas y confianza para una vista limpia
            result.export_visuals(
                export_dir=temp_web_dir, 
                file_name="preview_slicing", 
                hide_labels=True, 
                hide_conf=True
            )
            
            st.image(os.path.join(temp_web_dir, "preview_slicing.png"), use_container_width=True)

        # --- SECCIÓN DE GUARDADO ---
        st.write("---")
        if st.button("💾 Guardar Resultado"):
            output_dir = "outputs_racimos"
            os.makedirs(output_dir, exist_ok=True)
            
            name_only = os.path.splitext(uploaded_file.name)[0]
            final_name = f"det_racimo_{name_only}_conf_{int(conf_val*100)}"
            
            result.export_visuals(
                export_dir=output_dir, 
                file_name=final_name, 
                hide_labels=True
            )
            st.success(f"✅ Imagen guardada en: {output_dir}/{final_name}.png")

        st.info(f"Se han detectado {num_racimos} racimos usando Slicing Aided Hyper Inference (SAHI).")