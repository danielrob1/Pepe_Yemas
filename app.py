import streamlit as st
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
from PIL import Image

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Pepe IA", layout="wide")
st.title("🌿 Pepe IA: Contador de Yemas")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración")
conf_val = st.sidebar.select_slider(
    "Seleccionar Confidence (Umbral de confianza)",
    options=[round(i * 0.05, 2) for i in range(1, 20)],
    value=0.25,
    help="Recomendado: 0.25"
)
st.sidebar.caption(f"⬅️ + Falsos positivos | **{conf_val}** (Recomendado) | - Predicciones reales ➡️")

VIS_BASE_DIR = "visualizacion_dataset_completo"

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model(_conf):
    model_path = os.path.join('runs', 'detect', 'yemas_model_v2_augmentation', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        st.error(f" No se encontró el modelo en: {model_path}. Revisa la carpeta 'runs'.")
        st.stop()
        
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=_conf,
        device='cuda:0',
    )

model = load_model(conf_val)

# --- INTERFAZ DE SUBIDA ---
uploaded_file = st.file_uploader("Sube la imagen del viñedo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Guardar temporal para procesamiento
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    filename = uploaded_file.name
    name_only = os.path.splitext(filename)[0]
    
    with st.spinner("Procesando predicción..."):
        # 1. EJECUTAR SAHI
        result = get_sliced_prediction(
            temp_path,
            model,
            slice_height=800,
            slice_width=800,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        # 2. FILTRAR SOLO YEMAS
        result.object_prediction_list = [
            pred for pred in result.object_prediction_list if "yema" in pred.category.name.lower()
        ]

        # 3. LÓGICA DE BÚSQUEDA DE IMAGEN ETIQUETADA
        img_etiquetada_path = None
        for split in ["train", "val", "test"]:
            posible_ruta = os.path.join(VIS_BASE_DIR, split, filename)
            if os.path.exists(posible_ruta):
                img_etiquetada_path = posible_ruta
                break

        # --- MOSTRAR COMPARATIVA ---
        col1, col2 = st.columns(2)
        
        with col1:
            if img_etiquetada_path:
                st.subheader("📍 Etiquetas Manuales")
                st.image(img_etiquetada_path, use_container_width=True)
            else:
                st.subheader("🖼️ Imagen Original")
                st.image(uploaded_file, use_container_width=True)
        
        with col2:
            st.subheader(f"🤖 Predicción IA ({len(result.object_prediction_list)} yemas)")
            # Exportamos a una carpeta temporal solo para previsualizar en la web
            temp_output_dir = "temp_web"
            os.makedirs(temp_output_dir, exist_ok=True)
            result.export_visuals(export_dir=temp_output_dir, file_name="preview", hide_labels=True)
            st.image(os.path.join(temp_output_dir, "preview.png"), use_container_width=True)

        # --- BOTÓN DE GUARDADO ---
        st.write("---")
        if st.button("💾 Guardar Resultado en carpeta Outputs"):
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            # Guardamos con el nombre original y el umbral usado
            final_name = f"solo_yemas_{name_only}_conf_{int(conf_val*100)}"
            result.export_visuals(export_dir=output_dir, file_name=final_name, hide_labels=True)
            st.success(f"✅ Archivo guardado como: {final_name}.png en la carpeta {output_dir}")

        st.info(f"Yemas detectadas con confianza > {conf_val}: {len(result.object_prediction_list)}")