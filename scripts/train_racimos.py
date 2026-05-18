from ultralytics import YOLO

# Cargamos el modelo 'small' (s)
# Es un equilibrio perfecto para 500 imágenes: más capaz que el 'nano' 
# pero no tan grande como para memorizar (overfitting)
def train():
    model = YOLO('yolo26s.pt') 

    results = model.train(
        data='data_racimos.yaml',
        epochs=150,           # Subimos a 150 porque con mucha aumentación tarda más en aprender
        imgsz=640,           # Mantenemos resolución alta para detalles del racimo
        batch=16,             # Ajusta según tu hardware
        patience=30,          # Le damos margen para aprender con los datos "difíciles" aumentados
        name="racimos_sliced_yolo26_v1",
        save_period=1,
        
        # --- AUGMENTATION ESTRATÉGICA PARA DATASETS PEQUEÑOS ---
        mosaic=1.0,           # Obligatorio: combina 4 fotos en 1 para ver escalas
        mixup=0.2,            # Mezcla dos imágenes; vital para detectar uvas entre hojas
        degrees=15.0,         # Rotaciones para simular diferentes perspectivas de cámara
        shear=2.0,            # Distorsión de planos
        perspective=0.0001,   # Pequeño cambio de perspectiva 3D
        hsv_h=0.015,          # Cambios de tono (uvas más verdes o más oscuras)
        hsv_s=0.7,            # Cambios de saturación para días nublados vs soleados
        hsv_v=0.4,            # Cambios de brillo para manejar sombras en el viñedo
        fliplr=0.5,           # Volteo horizontal (duplica tus datos visualmente)
        
        # --- REGULARIZACIÓN (PARA EVITAR OVERFITTING) ---
        dropout=0.1,          # Apaga neuronas al azar para que no se memorice el dataset
        label_smoothing=0.1,  # Hace al modelo menos rígido con las etiquetas
        weight_decay=0.0005   # Penaliza la complejidad excesiva del modelo
    )

if __name__ == "__main__":
    train()