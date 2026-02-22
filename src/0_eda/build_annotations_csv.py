import os
import json
import pandas as pd

from urllib.parse import urlparse, parse_qs

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR   = os.path.join(BASE_DIR, "data")
META_DIR   = os.path.join(DATA_DIR, "meta")
ANN_DIR    = os.path.join(DATA_DIR, "annotations")

IMG_ANNOTATED_DIR = os.path.join(DATA_DIR, "images", "annotated_images")

LABEL_STUDIO_DIR = os.path.join(DATA_DIR, "label_studio")

os.makedirs(ANN_DIR, exist_ok=True)

LS_EXPORT_PATH = os.path.join(LABEL_STUDIO_DIR, "labelstudio_export.json")
OUTPUT_CSV     = os.path.join(ANN_DIR, "annotations.csv")

print("BASE_DIR:", BASE_DIR)
print("Leyendo Label Studio JSON de:", LS_EXPORT_PATH)

with open(LS_EXPORT_PATH, "r", encoding="utf-8") as f:
    tasks = json.load(f)

print("Número de tareas en el JSON:", len(tasks))

records = []

# Clases principales
MAIN_CLASSES = {"obstaculo", "no_obstaculo", "acera", "carretera"}

# Conjunto de imágenes válidas
annotated_images = {
    f for f in os.listdir(IMG_ANNOTATED_DIR)
    if os.path.isfile(os.path.join(IMG_ANNOTATED_DIR, f))
}


def extract_image_id_from_data_url(url_str: str) -> str:
    """
    En Label Studio la ruta viene como:
    '/data/local-files/?d=data/raw/gsv-amsterdam-1081-Obstacle.png'
    Nos quedamos con el nombre de archivo: 'gsv-amsterdam-1081-Obstacle.png'
    """
    parsed = urlparse(url_str)
    qs = parse_qs(parsed.query)
    if "d" in qs:
        rel_path = qs["d"][0]
        return os.path.basename(rel_path)
    return os.path.basename(parsed.path)


def select_valid_annotation(annotations):
    """
    Devuelve la anotación válida:
    - no cancelada (was_cancelled == False)
    - con result no vacío
    - si hay varias, la más reciente (updated_at)
    """
    valid = [
        a for a in annotations
        if (not a.get("was_cancelled", False)) and a.get("result")
    ]

    if not valid:
        return None

    valid.sort(key=lambda a: a.get("updated_at", ""), reverse=True)
    return valid[0]


for task in tasks:
    data = task.get("data", {})
    image_field = data.get("image") 
    if image_field is None:
        print(f"[AVISO] Tarea sin campo 'image': id={task.get('id')}")
        continue

    image_id = extract_image_id_from_data_url(image_field)

    if image_id not in annotated_images:
        continue

    annotations = task.get("annotations", [])
    if not annotations:
        continue

    ann = select_valid_annotation(annotations)
    if ann is None:
        continue

    results = ann["result"]

    objects = {}

    for r in results:
        r_type = r.get("type")
        if r_type != "rectanglelabels":
            continue

        obj_id = r.get("id")
        if obj_id is None:
            continue

        v = r.get("value", {})
        rect_labels = v.get("rectanglelabels", [])
        if not rect_labels:
            continue

        class_name = rect_labels[0]

        orig_w = r.get("original_width")
        orig_h = r.get("original_height")

        # Coordenadas en porcentaje (0-100)
        x_perc = v.get("x")
        y_perc = v.get("y")
        w_perc = v.get("width")
        h_perc = v.get("height")

        if None in (orig_w, orig_h, x_perc, y_perc, w_perc, h_perc):
            continue

        # Convertir porcentajes a píxeles
        xmin = x_perc / 100.0 * orig_w
        ymin = y_perc / 100.0 * orig_h
        xmax = (x_perc + w_perc) / 100.0 * orig_w
        ymax = (y_perc + h_perc) / 100.0 * orig_h

        objects[obj_id] = {
            "image_id": image_id,
            "image_width": orig_w,
            "image_height": orig_h,
            "class_name": class_name if class_name in MAIN_CLASSES else "otros",
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "tipo_obstaculo": "",
            "temporalidad": "",
            "visibilidad": "",
            "severidad": "",
            "anotacion": "",
            "altura": "",
        }

    for r in results:
        if r.get("type") != "choices":
            continue

        obj_id = r.get("id")
        if obj_id not in objects:
            continue

        v = r.get("value", {})
        choices = v.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        from_name = r.get("from_name")
        
        if from_name == "tipo_obstaculo":
            objects[obj_id]["tipo_obstaculo"] = choice
        elif from_name == "temporalidad":
            objects[obj_id]["temporalidad"] = choice
        elif from_name == "calidad_visibilidad":
            objects[obj_id]["visibilidad"] = choice
        elif from_name == "grado_obstaculizacion":
            objects[obj_id]["severidad"] = choice
        elif from_name == "validacion_anotacion":
            objects[obj_id]["anotacion"] = choice
        elif from_name == "alutra_obstaculo_centimetros":
            objects[obj_id]["altura"] = choice
        else:
            pass

    records.extend(objects.values())

# Construir DataFrame
df_ann = pd.DataFrame(records)
print("Número de objetos anotados:", len(df_ann))
print(df_ann.head())

df_ann.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print("Guardado annotations.csv en:", OUTPUT_CSV)