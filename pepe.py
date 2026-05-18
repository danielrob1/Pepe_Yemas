import streamlit as st
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import pandas as pd
import math
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Pepe IA: Contador de Yemaas", layout="wide")
st.title("Pepe IA: Contador de Yemas")

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Configuración")
modo = st.sidebar.radio("Seleccionar Modo:", ["Imagen Individual", "Procesamiento por Lotes (Batch)"])

conf_val = st.sidebar.select_slider(
    "Umbral de confianza",
    options=[round(i * 0.05, 2) for i in range(1, 20)],
    value=0.25
)

# Factores definitivos (Val + Test)
FACTOR_CORRECCION = 1.22
MARGEN_ERROR = 0.12

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model(_conf):
    model_path = os.path.join('runs', 'detect', 'yemas_new_model_V2_augmentation_final', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        st.error(f"❌ Modelo no encontrado en: {model_path}")
        st.stop()
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=_conf,
        device='cuda:0', # Cambiar a 'cpu' si no hay GPU disponible
    )

model = load_model(conf_val)

def calcular_estadisticas(n):
    estimacion = n * FACTOR_CORRECCION
    margen = estimacion * MARGEN_ERROR
    return {
        "estimacion": round(estimacion),
        "rango": f"{max(0, math.floor(estimacion - margen))} - {math.ceil(estimacion + margen)}"
    }

# --- MODO: IMAGEN INDIVIDUAL ---
if modo == "Imagen Individual":
    uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_path = "temp_input.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Pepe analizando..."):
            result = get_sliced_prediction(temp_path, model, slice_height=800, slice_width=800)
            result.object_prediction_list = [p for p in result.object_prediction_list if "yema" in p.category.name.lower()]
            
            n = len(result.object_prediction_list)
            stats = calcular_estadisticas(n)
            
            # Mostrar resultados (UI anterior simplificada)
            st.image(temp_path, caption="Imagen procesada", use_container_width=True)
            st.metric("Estimación Real", stats['estimacion'], delta=f"Rango: {stats['rango']}")

# --- MODO: PROCESAMIENTO POR LOTES ---
else:
    uploaded_files = st.file_uploader("Sube varias imágenes...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🚀 Iniciar Procesamiento por Lotes"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_folder = f"outputs/batch_{timestamp}"
            images_folder = os.path.join(batch_folder, "imagenes_marcadas")
            results_folder = os.path.join(batch_folder, "resultados_csv")
            
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(results_folder, exist_ok=True)
            
            data_list = []
            progreso = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                # Guardado temporal
                path = f"temp_{file.name}"
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Predicción
                result = get_sliced_prediction(path, model, slice_height=800, slice_width=800)
                result.object_prediction_list = [p for p in result.object_prediction_list if "yema" in p.category.name.lower()]
                
                n = len(result.object_prediction_list)
                stats = calcular_estadisticas(n)
                
                # Guardar imagen marcada
                result.export_visuals(export_dir=images_folder, file_name=f"marcada_{file.name}", hide_labels=True)
                
                # Añadir a la lista del CSV
                data_list.append({
                    "foto": file.name,
                    "deteccion": n,
                    "estimacion": stats['estimacion'],
                    "intervalo": stats['rango']
                })
                
                progreso.progress((i + 1) / len(uploaded_files))
                os.remove(path) # Limpiar temp

            # Crear DataFrame y Fila de Totales
            df = pd.DataFrame(data_list)
            totales = pd.DataFrame([{
                "foto": "TOTAL BATCH",
                "deteccion": df["deteccion"].sum(),
                "estimacion": df["estimacion"].sum(),
                "intervalo": "N/A"
            }])
            df_final = pd.concat([df, totales], ignore_index=True)
            
            # Guardar CSV
            csv_path = os.path.join(results_folder, f"reporte_{timestamp}.csv")
            df_final.to_csv(csv_path, index=False)
            
            st.success(f"✅ Procesamiento completado. Archivos guardados en: {batch_folder}")
            st.table(df_final)