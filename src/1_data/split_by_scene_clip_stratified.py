import os
import re
import torch
import shutil
import random
import open_clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_distances

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR  = os.path.join(DATA_DIR, "images", "annotated_images")
LBL_DIR  = os.path.join(DATA_DIR, "labels")

OUT_IMG   = os.path.join(DATA_DIR, "images")
TRAIN_DIR = os.path.join(OUT_IMG, "train")
VAL_DIR   = os.path.join(OUT_IMG, "val")
TEST_DIR  = os.path.join(OUT_IMG, "test")

META_DIR   = os.path.join(DATA_DIR, "meta")
REPORT_CSV = os.path.join(DATA_DIR, "annotations", "split_report.csv")

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, META_DIR]:
    os.makedirs(d, exist_ok=True)

# Configuraciones
N_VAL  = 125
N_TEST = 125
SEED = 42

# Definicion de los parametros
WINDOW = 15
DIST_THRESHOLD = 0.15
MAX_SCENE_LEN = 80

# Estratifiación
OBSTACLE_CLASS_ID = 0
POS_RATIO = 0.7

# Balanceo por ciudad
# Mínimo de imágenes por ciudad
MIN_CITY_IMAGES_FOR_ENFORCEMENT = 40
# Overshoot permitido en imágenes
MAX_OVERSHOOT = 15

# Parametros CLIP
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# I/O
DRY_RUN = False
IMG_EXTS = {".jpg", ".jpeg", ".png"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Helpers
CITY_RE = re.compile(r"^gsv-([^-]+)-", re.IGNORECASE)

def extract_city(filename: str) -> str:
    """
    Espera nombres tipo: gsv-cdmx-54442-Obstacle.png
    Devuelve 'cdmx'. Si no matchea, 'unknown'.
    """
    m = CITY_RE.match(filename)
    return m.group(1).lower() if m else "unknown"

def yolo_label_path(img_name: str) -> str:
    # Soporta .jpg/.png -> .txt
    stem = os.path.splitext(img_name)[0]
    return os.path.join(LBL_DIR, stem + ".txt")

def image_has_obstacle(img_name: str) -> bool:
    p = yolo_label_path(img_name)
    if not os.path.exists(p):
        return False
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if int(parts[0]) == OBSTACLE_CLASS_ID:
                return True
    return False

# Cargar imnagenes
img_paths = sorted(
    [p for p in Path(IMG_DIR).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
    key=lambda p: p.name
)
names = [p.name for p in img_paths]
cities = [extract_city(n) for n in names]

print(f"Images loaded: {len(img_paths)} | Device: {DEVICE}")
print("Example:", names[0], "-> city:", cities[0])

# Cargar CLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(DEVICE).eval()

def load_img(p: Path):
    return preprocess(Image.open(p).convert("RGB"))

# Calcular embeddings
embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(img_paths), BATCH_SIZE), desc="CLIP embeddings"):
        batch = img_paths[i:i+BATCH_SIZE]
        imgs = torch.stack([load_img(p) for p in batch]).to(DEVICE)
        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())

embeddings = np.vstack(embeddings).astype(np.float32)

#  Clustering por escenas 
scenes = []
current = [0]

for i in range(1, len(embeddings)):
    prev = current[-WINDOW:]
    d = cosine_distances(embeddings[i:i+1], embeddings[prev]).min()

    if d <= DIST_THRESHOLD and len(current) < MAX_SCENE_LEN:
        current.append(i)
    else:
        scenes.append(current)
        current = [i]

scenes.append(current)

print(f"Scenes detected: {len(scenes)}")
print("Top scene sizes:", sorted([len(s) for s in scenes], reverse=True)[:10])

scene_meta = []
for sid, idxs in enumerate(scenes):
    scene_cities = [cities[i] for i in idxs]
    city = Counter(scene_cities).most_common(1)[0][0]

    has_obstacle = any(image_has_obstacle(names[i]) for i in idxs)

    scene_meta.append({
        "scene_id": sid,
        "indices": idxs,
        "n_imgs": len(idxs),
        "has_obstacle": bool(has_obstacle),
        "city": city,
    })

# Conteo de imágenes por ciudad
city_img_counts = Counter(cities)

# Ciudades suficientemente grandes para forzar representación
eligible_cities = {
    c for c, n in city_img_counts.items()
    if n >= MIN_CITY_IMAGES_FOR_ENFORCEMENT
}

print(f"Eligible cities for enforced coverage (>= {MIN_CITY_IMAGES_FOR_ENFORCEMENT} imgs): {sorted(list(eligible_cities))[:15]} ...")
print(f"Total eligible cities: {len(eligible_cities)}")

# Calcular por ciudad  para val/test
def compute_city_quotas(target_imgs: int) -> dict:
    """
    Cuotas proporcionales por ciudad (solo para ciudades elegibles),
    redondeadas y ajustadas para sumar target_imgs (aprox, porque luego seleccionamos escenas).
    """
    total_eligible_imgs = sum(city_img_counts[c] for c in eligible_cities) or 1
    raw = {c: (city_img_counts[c] / total_eligible_imgs) * target_imgs for c in eligible_cities}

    # redondeo inicial
    quotas = {c: int(round(v)) for c, v in raw.items()}

    # asegurar mínimo 1 para ciudades elegibles no minúsculas
    for c in eligible_cities:
        if quotas[c] == 0 and target_imgs >= len(eligible_cities):
            quotas[c] = 1

    # ajustar suma
    diff = target_imgs - sum(quotas.values())
    if diff != 0:
        frac = sorted(raw.items(), key=lambda kv: (kv[1] - int(kv[1])), reverse=(diff > 0))
        k = 0
        while diff != 0 and k < 100000:
            c = frac[k % len(frac)][0]
            if diff > 0:
                quotas[c] += 1
                diff -= 1
            else:
                if quotas[c] > 0:
                    quotas[c] -= 1
                    diff += 1
            k += 1

    return quotas

test_city_quota = compute_city_quotas(N_TEST)
val_city_quota  = compute_city_quotas(N_VAL)

# Estratificación balanceada por escena
def pick_scenes_balanced_best_fit(
    target_imgs: int,
    scenes_pool: list,
    city_quota: dict,
    pos_ratio: float,
    tries: int = 4000,
    max_overshoot: int = 15,
):
    """
    Selecciona escenas completas para aproximar target_imgs,
    cumpliendo:
      - ratio aproximado de escenas con obstáculo (POS_RATIO)
      - cuotas por ciudad (aprox, a nivel imágenes)
    Estrategia: random search con score multi-objetivo.
    """
    best = None
    best_score = float("inf")

    pool_pos = [s for s in scenes_pool if s["has_obstacle"]]
    pool_neg = [s for s in scenes_pool if not s["has_obstacle"]]

    for _ in range(tries):
        random.shuffle(pool_pos)
        random.shuffle(pool_neg)

        chosen = []
        total = 0
        pos_target = int(target_imgs * pos_ratio)

        city_used_imgs = defaultdict(int)
        pos_imgs = 0

        def can_add(scene):
            c = scene["city"]
            if c in city_quota:
                pass
            return True

        for s in pool_pos:
            if pos_imgs >= pos_target:
                break
            if total + s["n_imgs"] > target_imgs + max_overshoot:
                continue
            if not can_add(s):
                continue
            chosen.append(s)
            total += s["n_imgs"]
            pos_imgs += s["n_imgs"]
            city_used_imgs[s["city"]] += s["n_imgs"]

        for s in pool_neg:
            if total >= target_imgs:
                break
            if total + s["n_imgs"] > target_imgs + max_overshoot:
                continue
            if not can_add(s):
                continue
            chosen.append(s)
            total += s["n_imgs"]
            city_used_imgs[s["city"]] += s["n_imgs"]

        if total < target_imgs:
            pool_all = pool_pos + pool_neg
            random.shuffle(pool_all)
            for s in pool_all:
                if s in chosen:
                    continue
                if total >= target_imgs:
                    break
                if total + s["n_imgs"] > target_imgs + max_overshoot:
                    continue
                chosen.append(s)
                total += s["n_imgs"]
                if s["has_obstacle"]:
                    pos_imgs += s["n_imgs"]
                city_used_imgs[s["city"]] += s["n_imgs"]

        overshoot = max(0, total - target_imgs)

        # Score multi-objetivo
        score = abs(total - target_imgs)

        # penalizar overshoot grande
        if overshoot > max_overshoot:
            score += 1000

        # penalización por desviación del ratio pos
        desired_pos = pos_target
        score += 0.5 * abs(pos_imgs - desired_pos)

        # penalización por desviación de cuotas por ciudad
        for c, q in city_quota.items():
            used = city_used_imgs.get(c, 0)
            score += 0.3 * abs(used - q)

        # penalización fuerte si una ciudad elegible no aparece nada
        for c in city_quota.keys():
            if city_used_imgs.get(c, 0) == 0:
                score += 50

        if score < best_score and total > 0:
            best_score = score
            best = chosen

        if best_score == 0:
            break

    return best


# Split
# pool inicial
pool = scene_meta[:]

test_scenes = pick_scenes_balanced_best_fit(
    N_TEST, pool, test_city_quota, POS_RATIO,
    tries=5000, max_overshoot=MAX_OVERSHOOT
)
used_ids = set(s["scene_id"] for s in test_scenes)

remaining = [s for s in pool if s["scene_id"] not in used_ids]

val_scenes = pick_scenes_balanced_best_fit(
    N_VAL, remaining, val_city_quota, POS_RATIO,
    tries=5000, max_overshoot=MAX_OVERSHOOT
)
used_ids |= set(s["scene_id"] for s in val_scenes)

train_scenes = [s for s in pool if s["scene_id"] not in used_ids]

def flatten(scene_list):
    return [i for s in scene_list for i in s["indices"]]

test_idxs  = flatten(test_scenes)
val_idxs   = flatten(val_scenes)
train_idxs = flatten(train_scenes)

print(f"FINAL SPLIT -> Train:{len(train_idxs)} | Val:{len(val_idxs)} | Test:{len(test_idxs)}")


def copy_images(idxs, dst):
    for i in idxs:
        src = img_paths[i]
        dst_path = os.path.join(dst, src.name)
        if not os.path.exists(dst_path) and not DRY_RUN:
            shutil.copy2(src, dst_path)

copy_images(train_idxs, TRAIN_DIR)
copy_images(val_idxs, VAL_DIR)
copy_images(test_idxs, TEST_DIR)

split = [""] * len(names)
for i in train_idxs: split[i] = "train"
for i in val_idxs:   split[i] = "val"
for i in test_idxs:  split[i] = "test"

scene_id_per_image = np.empty(len(names), dtype=int)
has_obstacle_per_image = np.zeros(len(names), dtype=int)

for s in scene_meta:
    for i in s["indices"]:
        scene_id_per_image[i] = s["scene_id"]
        has_obstacle_per_image[i] = int(s["has_obstacle"])

df = pd.DataFrame({
    "image": names,
    "split": split,
    "scene_id": scene_id_per_image,
    "city": cities,
    "has_obstacle_scene": has_obstacle_per_image,
})

df.to_csv(REPORT_CSV, index=False)
print(f"Split report saved to {REPORT_CSV}")
print(f"DRY_RUN={DRY_RUN}")