import streamlit as st
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
from PIL import Image
import math

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Pepe IA: Contador de Yemas", layout="wide")
st.title("Pepe IA: Contador de Yemas")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración")
conf_val = st.sidebar.select_slider(
    "Seleccionar Confidence (Umbral de confianza)",
    options=[round(i * 0.05, 2) for i in range(1, 20)],
    value=0.25,
    help="Recomendado: 0.25"
)

VIS_BASE_DIR = "visualizacion_dataset_completo"

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model(_conf):
    model_path = os.path.join('runs', 'detect', 'yemas_new_model_V2_augmentation_final', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        st.error(f"❌ No se encontró el modelo en: {model_path}.")
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
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    filename = uploaded_file.name
    name_only = os.path.splitext(filename)[0]
    
    with st.spinner("Pepe está analizando y calculando estimaciones..."):
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
        
        num_detectado = len(result.object_prediction_list)

        # --- 3. CÁLCULOS ESTADÍSTICOS ---
        # Factor de corrección: 1.22 
        # Margen de error: 12%
        valor_real_esperado = num_detectado * 1.22
        margen = valor_real_esperado * 0.12
        
        limite_inferior = max(0, math.floor(valor_real_esperado - margen))
        limite_superior = math.ceil(valor_real_esperado + margen)
        centro_redondeado = round(valor_real_esperado)

        # --- MOSTRAR COMPARATIVA ---
        col1, col2 = st.columns(2)
        
        with col1:
            img_etiquetada_path = None
            for split in ["train", "val", "test"]:
                posible_ruta = os.path.join(VIS_BASE_DIR, split, filename)
                if os.path.exists(posible_ruta):
                    img_etiquetada_path = posible_ruta
                    break

            if img_etiquetada_path:
                st.subheader("Etiquetas Manuales (Referencia)")
                st.image(img_etiquetada_path, use_container_width=True)
            else:
                st.subheader("Imagen Original")
                st.image(uploaded_file, use_container_width=True)
        
        with col2:
            st.subheader(f" Detección de Pepe: {num_detectado} yemas")
            temp_output_dir = "temp_web"
            os.makedirs(temp_output_dir, exist_ok=True)
            result.export_visuals(export_dir=temp_output_dir, file_name="preview", hide_labels=True)
            st.image(os.path.join(temp_output_dir, "preview.png"), use_container_width=True)

        # --- 📊 SECCIÓN DE RESULTADOS ESTADÍSTICOS ---
        st.markdown("---")
        st.header("📊 Informe de Estimación")
        
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric("Detectadas", f"{num_detectado}")
            st.caption("Conteo bruto realizado por Pepe.")

        with m2:
            st.metric("Valor Real Estimado", f"~{centro_redondeado}")
            st.caption("Ajustado por el factor de corrección (1.22).")

        with m3:
            st.metric("Intervalo de Confianza", f"{limite_inferior} - {limite_superior}")
            st.caption("Rango probable basado en un margen del 12%.")

        # --- BOTÓN DE GUARDADO ---
        if st.button("💾 Guardar Resultado en carpeta Outputs"):
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            final_name = f"det_{name_only}_conf_{int(conf_val*100)}"
            result.export_visuals(export_dir=output_dir, file_name=final_name, hide_labels=True)
            st.success(f"✅ Archivo guardado. Estimación: {centro_redondeado} (Rango: {limite_inferior}-{limite_superior})")