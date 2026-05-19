import os
import cv2
import random

def leer_labels(label_path):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f.readlines():
            values = line.strip().split()

            if len(values) < 5:
                continue  # ignorar líneas inválidas

            values = list(map(float, values))
            cls = values[0]

            try:
                # caso bbox normal
                if len(values) == 5:
                    x, y, w, h = values[1:5]

                # caso segmentación
                else:
                    coords = values[1:]
                    xs = coords[::2]
                    ys = coords[1::2]

                    if len(xs) == 0 or len(ys) == 0:
                        continue

                    x_min = min(xs)
                    x_max = max(xs)
                    y_min = min(ys)
                    y_max = max(ys)

                    x = (x_min + x_max) / 2
                    y = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min

                # validar valores
                if w <= 0 or h <= 0:
                    continue

                boxes.append((cls, x, y, w, h))

            except:
                continue  # ignora cualquier cosa rara

    return boxes



def guardar_labels(path, boxes):
    with open(path, "w") as f:
        for b in boxes:
            f.write(" ".join(map(str, b)) + "\n")

#Resolucion baja 512, 256
#Resolucion alta 800, 400
def slice_image(img, boxes, size=800, stride=400):
    h, w, _ = img.shape
    slices = []

    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):

            crop = img[y:y+size, x:x+size]
            new_boxes = []

            for box in boxes:
                # validar formato
                if len(box) != 5:
                    continue

                cls, bx, by, bw, bh = box

                # evitar valores raros
                if bw <= 0 or bh <= 0:
                    continue

                # convertir a píxeles
                cx = bx * w
                cy = by * h
                bw_px = bw * w
                bh_px = bh * h

                x1 = cx - bw_px / 2
                y1 = cy - bh_px / 2
                x2 = cx + bw_px / 2
                y2 = cy + bh_px / 2

                # calcular intersección con el crop
                inter_x1 = max(x1, x)
                inter_y1 = max(y1, y)
                inter_x2 = min(x2, x + size)
                inter_y2 = min(y2, y + size)

                inter_w = inter_x2 - inter_x1
                inter_h = inter_y2 - inter_y1

                # si no hay intersección → ignorar
                if inter_w <= 0 or inter_h <= 0:
                    continue

                # mantener solo si suficiente dentro del crop
                area_original = bw_px * bh_px
                area_inter = inter_w * inter_h

                if area_inter / area_original < 0.3:
                    continue

                # convertir a coords relativas al crop
                new_cx = ((inter_x1 + inter_x2) / 2 - x) / size
                new_cy = ((inter_y1 + inter_y2) / 2 - y) / size
                new_bw = inter_w / size
                new_bh = inter_h / size

                # evitar cajas absurdas
                if new_bw <= 0 or new_bh <= 0:
                    continue

                new_boxes.append((cls, new_cx, new_cy, new_bw, new_bh))

            if len(new_boxes) > 0 or random.random() < 0.1:
                slices.append((crop, new_boxes))

    return slices


def procesar_split(split):
    img_dir = f"dataset_racimos_v2/images/{split}"
    label_dir = f"dataset_racimos_v2/labels/{split}"

    out_img_dir = f"sliced_dataset_racimos_v3/{split}/images"
    out_lbl_dir = f"sliced_dataset_racimos_v3/{split}/labels"

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        if not img_name.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        boxes = leer_labels(label_path)

        slices = slice_image(img, boxes)

        base = os.path.splitext(img_name)[0]

        for i, (crop, new_boxes) in enumerate(slices):
            out_img = f"{base}_{i}.jpg"
            out_lbl = f"{base}_{i}.txt"

            cv2.imwrite(os.path.join(out_img_dir, out_img), crop)
            guardar_labels(os.path.join(out_lbl_dir, out_lbl), new_boxes)

    print(f"✅ {split} procesado")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        procesar_split(split)