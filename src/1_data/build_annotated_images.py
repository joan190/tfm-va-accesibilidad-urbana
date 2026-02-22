import os
import json
import shutil
import random

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR   = os.path.join(BASE_DIR, "data")

META_DIR   = os.path.join(DATA_DIR, "meta")

ALL_IMG_DIR = os.path.join(DATA_DIR, "images", "all")
ANNOTATED_IMG_DIR = os.path.join(DATA_DIR, "images", "annotated_images")

LS_EXPORT_PATH = os.path.join(DATA_DIR, "label_studio", "labelstudio_export.json")

os.makedirs(ANNOTATED_IMG_DIR, exist_ok=True)

# Config
REQUIRED_CLASS = "acera" 
MAX_TO_COPY = 1250        
SEED = 42                
DRY_RUN = False        

# Helpers
def extract_labels_from_result_item(result_item: dict) -> set[str]:
    """
    Extrae labels de un item típico de Label Studio result[].
    Cubre casos comunes:
      - rectanglelabels: ["acera", ...]
      - polygonlabels:   ["acera", ...]
      - labels:          ["acera", ...]
      - choices:         ["acera", ...]
    """
    labels = set()
    value = result_item.get("value", {}) or {}

    for key in ("rectanglelabels", "polygonlabels", "labels", "choices"):
        v = value.get(key)
        if isinstance(v, list):
            labels.update([str(x) for x in v])

    for key in ("label", "labels"):
        v = result_item.get(key)
        if isinstance(v, list):
            labels.update([str(x) for x in v])
        elif isinstance(v, str):
            labels.add(v)

    return labels

def task_has_required_class(task: dict, required_class: str) -> bool:
    annotations = task.get("annotations", []) or []
    for ann in annotations:
        results = ann.get("result", []) or []
        for r in results:
            labels = extract_labels_from_result_item(r)
            if required_class in labels:
                return True
    return False

# Cargar el export de Label Studio
with open(LS_EXPORT_PATH, "r", encoding="utf-8") as f:
    tasks = json.load(f)

images_to_copy = set()

for task in tasks:
    annotations = task.get("annotations", []) or []
    if not annotations:
        continue

    has_results = any((ann.get("result") or []) for ann in annotations)
    if not has_results:
        continue

    if not task_has_required_class(task, REQUIRED_CLASS):
        continue

    image_path = (
        task.get("data", {}).get("image") or
        task.get("image")
    )
    if image_path:
        image_name = os.path.basename(image_path)
        images_to_copy.add(image_name)

images_to_copy = list(images_to_copy)
print(f"Encontradas con anotación + clase '{REQUIRED_CLASS}': {len(images_to_copy)}")

if len(images_to_copy) > MAX_TO_COPY:
    random.seed(SEED)
    images_to_copy = random.sample(images_to_copy, MAX_TO_COPY)
    print(f"Limitando a {MAX_TO_COPY} imágenes (SEED={SEED})")

copied = 0
missing = 0
skipped_exists = 0

for image_name in images_to_copy:
    src = os.path.join(ALL_IMG_DIR, image_name)
    dst = os.path.join(ANNOTATED_IMG_DIR, image_name)

    if not os.path.exists(src):
        missing += 1
        continue

    if os.path.exists(dst):
        skipped_exists += 1
        continue

    if not DRY_RUN:
        shutil.copy2(src, dst)

    copied += 1

print(f"Copiadas: {copied}")
print(f"Ya existían en destino (skip): {skipped_exists}")
if missing:
    print(f"No encontradas en images/all: {missing}")

print(f"Proceso terminado ✔ (DRY_RUN={DRY_RUN})")
