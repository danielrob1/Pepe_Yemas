import os
import random
import shutil

# --- CONFIGURACIÓN ---
base_path = "train"  # Tu carpeta actual
new_base = "dataset_racimos_v2" # La carpeta nueva que vamos a crear

images_source = os.path.join(base_path, "images")
labels_source = os.path.join(base_path, "labels")

# Proporciones
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

def organizar():
    # 1. Verificar que existen las carpetas de origen
    if not os.path.exists(images_source):
        print(f"❌ Error: No se encuentra la carpeta {images_source}")
        return

    # 2. Recolectar nombres de archivos (sin extensión)
    # Listamos archivos y quitamos la extensión para emparejar imagen con label
    all_files = [os.path.splitext(f)[0] for f in os.listdir(images_source) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    random.shuffle(all_files)

    # 3. Calcular puntos de corte
    total = len(all_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:]
    }

    # 4. Crear la nueva estructura de carpetas
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(new_base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(new_base, "labels", split), exist_ok=True)

    print(f"🚀 Organizando {total} archivos en '{new_base}'...")

    # 5. Copiar archivos a sus nuevos destinos
    for split, files in splits.items():
        for f_name in files:
            # Buscar la extensión original de la imagen
            img_ext = ""
            for ext in [".jpg", ".JPG", ".jpeg", ".png"]:
                if os.path.exists(os.path.join(images_source, f_name + ext)):
                    img_ext = ext
                    break
            
            if img_ext:
                # Rutas de origen
                old_img = os.path.join(images_source, f_name + img_ext)
                old_lbl = os.path.join(labels_source, f_name + ".txt")

                # Rutas de destino
                new_img = os.path.join(new_base, "images", split, f_name + img_ext)
                new_lbl = os.path.join(new_base, "labels", split, f_name + ".txt")

                # Copiamos (shutil.copy) para no borrar los originales por si acaso
                shutil.copy(old_img, new_img)
                if os.path.exists(old_lbl):
                    shutil.copy(old_lbl, new_lbl)

    print("\n✅ Proceso completado con éxito:")
    for s in splits:
        print(f" - {s.capitalize()}: {len(splits[s])} imágenes -> {new_base}/images/{s}/")

if __name__ == "__main__":
    organizar()