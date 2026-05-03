from ultralytics import YOLO

def train():
    # Usamos YOLO11 para una mejor arquitectura de extracción de características
    model = YOLO("yolo11s.pt") 

    model.train(
        data="data.yaml",
        epochs=150,           # Subimos a 150 para que le dé tiempo a aprender los nudos
        imgsz=1280,           # 1280 para entrenar con alta resolución
        batch=-1,             # 'batch=-1' hará que YOLO use el potencial optimo de la GPU
        name="yemas_model_v2_augmentation",
        device="0",           # Usar la GPU para el entrenamiento
        
        # --- MEJORAS PARA NUDOS Y DETALLE ---
        cls=2.0,              # Aumentamos el peso de la clasificación (evita confundir yema/nudo)
        box=7.5,              # Ajustamos la precisión de las cajas
        patience=40,          # Si deja de mejorar en 40 épocas, se detiene solo
        
        # --- DATA AUGMENTATION (Anti-ruido y sombras) ---
        hsv_v=0.4,            # Variación de brillo (para las sombras de tus fotos)
        hsv_s=0.5,            # Variación de saturación (para los tonos marrones)
        degrees=15.0,         # Rotaciones leves para que aprenda nudos desde varios ángulos
        mosaic=1.0,           # Obligatorio para objetos pequeños
        mixup=0.1,            # Mezcla imágenes para que no se aprenda de memoria el fondo
        close_mosaic=10       # Desactiva el mosaico al final para "afinar" el ojo en los detalles
    )

if __name__ == "__main__":
    train()