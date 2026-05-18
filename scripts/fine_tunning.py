from ultralytics import YOLO
import os

def train_fine_tuning():
    # 1. CARGA DEL MODELO PREVIO
    # Usamos los mejores pesos del entrenamiento que ya tienes
    path_modelo_previo = os.path.join('runs', 'detect', 'yemas_new_model_v2_augmentation_final', 'weights', 'best.pt')
    
    if not os.path.exists(path_modelo_previo):
        print(f"ERROR: No se encontró el modelo en {path_modelo_previo}")
        return

    model = YOLO(path_modelo_previo)

    # 2. CONFIGURACIÓN DEL ENTRENAMIENTO (Fine-Tuning de Sensibilidad)
    model.train(
        # Datos y nombre del experimento
        data="data.yaml",
        epochs=100,
        name="yemas_new_model_v2_augmentation_final_fine_tunned_2", # Crea una carpeta nueva automáticamente
        exist_ok=False,
        imgsz=1280,                       # Obliga a crear una carpeta nueva (v2, v3...)
        
        # Hardware y optimización
        batch=4,                              # Auto-ajuste de batch según memoria GPU
        device="0",
        workers=8,
        
        # --- ESTRATEGIA DE SENSIBILIDAD (Para ver las yemas que "no ve") ---
        lr0=0.0005,           # Learning rate muy bajo para no "romper" lo que ya sabe
        lrf=0.01,             # Factor final de aprendizaje
        warmup_epochs=3.0,    # Estabilización inicial
        
        # --- PREVENCIÓN DE SOBREAJUSTE ---
        patience=40,          
        save=True,
        save_period=1
    )

if __name__ == "__main__":
    train_fine_tuning()