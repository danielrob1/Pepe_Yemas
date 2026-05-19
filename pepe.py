import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import math
import csv
import gc
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Pepe IA: Contador de Yemas", layout="wide")
st.title(" Pepe IA: Contador de Yemas")

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Configuración")
modo = st.sidebar.radio("Seleccionar Modo:", ["Imagen Individual", "Procesamiento en masa"])

conf_val = st.sidebar.select_slider(
    "Seleccionar Confidence (Umbral de confianza)",
    options=[round(i * 0.05, 2) for i in range(1, 20)],
    value=0.25,
    help="Recomendado: 0.25"
)

VIS_BASE_DIR = "visualizacion_dataset_completo"
FACTOR_CORRECCION = 1.22
MARGEN_ERROR = 0.12

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

def calcular_estadisticas(n):
    estimacion = n * FACTOR_CORRECCION
    margen = estimacion * MARGEN_ERROR
    return {
        "estimacion": round(estimacion),
        "rango": f"{max(0, math.floor(estimacion - margen))} - {math.ceil(estimacion + margen)}"
    }

# --- MODO: IMAGEN INDIVIDUAL----
if modo == "Imagen Individual":
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
                overlap_width_ratio=0.2,
                verbose=0
            )

            # 2. FILTRAR SOLO YEMAS
            result.object_prediction_list = [
                pred for pred in result.object_prediction_list if "yema" in pred.category.name.lower()
            ]
            
            num_detectado = len(result.object_prediction_list)

            # 3. CÁLCULOS ESTADÍSTICOS
            stats = calcular_estadisticas(num_detectado)
            centro_redondeado = stats["estimacion"]
            limite_inferior = int(stats["rango"].split(" - ")[0])
            limite_superior = int(stats["rango"].split(" - ")[1])

            # --- MOSTRAR COMPARATIVA DE COLUMNAS ---
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
                st.caption(f"Ajustado por el factor de corrección ({FACTOR_CORRECCION}).")
            with m3:
                st.metric("Intervalo de Confianza", stats["rango"])
                st.caption(f"Rango probable basado en un margen del {int(MARGEN_ERROR*100)}%.")

            # --- BOTÓN DE GUARDADO ---
            if st.button("💾 Guardar Resultado en carpeta Outputs"):
                output_dir = "outputs"
                os.makedirs(output_dir, exist_ok=True)
                final_name = f"det_{name_only}_conf_{int(conf_val*100)}"
                result.export_visuals(export_dir=output_dir, file_name=final_name, hide_labels=True)
                st.success(f"✅ Archivo guardado. Estimación: {centro_redondeado} (Rango: {stats['rango']})")
            
            # Limpieza estable de memoria RAM/VRAM
            del result
            gc.collect()
            if os.path.exists(temp_path): os.remove(temp_path)

# --- MODO: PROCESAMIENTO POR LOTES ---
else:
    uploaded_files = st.file_uploader("Sube varias imágenes del viñedo...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Iniciar procesamiento en masa"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_folder = f"outputs/batch_{timestamp}"
        images_folder = os.path.join(batch_folder, "imagenes_marcadas")
        results_folder = os.path.join(batch_folder, "resultados_csv")
        
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)
        
        data_list = []
        num_fotos = len(uploaded_files)
        progreso = st.progress(0)
        status_text = st.empty()
        
        total_deteccion = 0
        total_estimacion = 0
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Procesando {i+1}/{num_fotos}: {file.name}")
            
            path = f"temp_batch_{file.name}"
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            
            result = get_sliced_prediction(path, model, slice_height=800, slice_width=800, verbose=0)
            result.object_prediction_list = [p for p in result.object_prediction_list if "yema" in p.category.name.lower()]
            
            n = len(result.object_prediction_list)
            stats = calcular_estadisticas(n)
            
            result.export_visuals(export_dir=images_folder, file_name=f"marcada_{file.name}", hide_labels=True)
            
            total_deteccion += n
            total_estimacion += stats['estimacion']
            
            data_list.append([file.name, n, stats['estimacion'], stats['rango']])
            
            del result
            gc.collect()
            if os.path.exists(path): os.remove(path)
            
            progreso.progress((i + 1) / num_fotos)
            
        status_text.text("💾 Generando reportes...")
        
        promedio_det = round(total_deteccion / num_fotos, 2)
        promedio_est = round(total_estimacion / num_fotos, 2)
        
        margen_total = total_estimacion * MARGEN_ERROR
        rango_total = f"{max(0, math.floor(total_estimacion - margen_total))} - {math.ceil(total_estimacion + margen_total)}"
        
        # Guardado del reporte
        csv_path = os.path.join(results_folder, f"reporte_{timestamp}.csv")
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["foto", "deteccion", "estimacion", "intervalo"])
            writer.writerows(data_list)
            writer.writerow(["TOTAL", total_deteccion, total_estimacion, rango_total])
            writer.writerow(["PROMEDIO POR IMAGEN", promedio_det, promedio_est, "-"])
            
        st.success(f"✅ Procesamiento en masa completado. Guardado en: {batch_folder}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Estimado", f"{total_estimacion}", delta=f"Rango: {rango_total}", delta_color="normal")
        col2.metric("Promedio Yemas/Foto", promedio_est)
        col3.metric("Imágenes Procesadas", num_fotos)
        
        st.markdown("### 📋 Resultados del Lote")
        md_tabla = "| Foto | Detección | Estimación | Intervalo |\n| :--- | :--- | :--- | :--- |\n"
        for fila in data_list:
            md_tabla += f"| {fila[0]} | {fila[1]} | **{fila[2]}** | {fila[3]} |\n"
        
        md_tabla += f"| **TOTAL** | **{total_deteccion}** | **{total_estimacion}** | **{rango_total}** |\n"
        md_tabla += f"| **PROMEDIO POR IMAGEN** | **{promedio_det}** | **{promedio_est}** | **-** |\n"
        
        st.markdown(md_tabla)