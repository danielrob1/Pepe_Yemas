from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import tkinter as tk
from tkinter import filedialog

# --- CONFIGURACIÓN DEL MODELO ---
model_path = os.path.join('runs', 'detect', 'yemas_model_v2_augmentation', 'weights', 'best.pt')

print("Cargando modelo en la GPU...")
model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.25, # Subimos un poco para filtrar cuadros azules/ruido
    device='cuda:0',
)

# --- SELECCIÓN DE ARCHIVO ---
root = tk.Tk()
root.withdraw() 
image_path = filedialog.askopenfilename(
    title="Seleccionar imagen",
    filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
)

if not image_path:
    exit()

# --- PROCESAMIENTO ---
filename = os.path.splitext(os.path.basename(image_path))[0]

print(f"Procesando: {filename}...")
result = get_sliced_prediction(
    image_path,
    model,
    slice_height=800,
    slice_width=800,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# --- FILTRO SOLO YEMAS (LO JUSTO Y NECESARIO) ---
# Sobreescribimos la lista para quedarnos SOLO con la categoría 'yema'
result.object_prediction_list = [
    pred for pred in result.object_prediction_list if "yema" in pred.category.name.lower()
]

# --- RESULTADOS Y GUARDADO ---
num_yemas = len(result.object_prediction_list)
print(f"✅ Total de yemas detectadas: {num_yemas}")

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

output_name = f"solo_yemas_{filename}"
result.export_visuals(export_dir=output_dir, file_name=output_name, hide_labels=True)

print(f"📁 Imagen guardada en: {output_dir}/{output_name}.png")