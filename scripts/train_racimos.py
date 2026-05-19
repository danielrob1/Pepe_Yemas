from ultralytics import YOLO

# Cargamos el modelo 'small' (s)
# Es un equilibrio perfecto para 500 imágenes: más capaz que el 'nano' 
# pero no tan grande como para memorizar (overfitting)
def train():
    model = YOLO('yolo26s.pt') 

    results = model.train(
        data='data.yaml',
        epochs=150,           # Subimos a 150 porque con mucha aumentación tarda más en aprender
        imgsz=640,           # Mantenemos resolución alta para detalles del racimo
        batch=16,             # Ajusta según tu hardware
        patience=30,          # Le damos margen para aprender con los datos "difíciles" aumentados
        name="racimos_sliced_v3_b",
        save_period=1,
        
        # --- AUGMENTATION ESTRATÉGICA PARA DATASETS PEQUEÑOS ---
        
        # --- REGULARIZACIÓN (PARA EVITAR OVERFITTING) ---
        dropout=0.1,          # Apaga neuronas al azar para que no se memorice el dataset
        label_smoothing=0.1,  # Hace al modelo menos rígido con las etiquetas
        weight_decay=0.0005   # Penaliza la complejidad excesiva del modelo
    )

if __name__ == "__main__":
    train()