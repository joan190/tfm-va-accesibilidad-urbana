# Pipeline de ejecución (TFM – Accesibilidad urbana con Visión Artificial)

Este documento describe el **orden recomendado de ejecución** del proyecto: **EDA → preprocesado → experimentos → evaluación**.

Orientado a **Windows (CMD / Anaconda Prompt)**.  
Si usas PowerShell o Linux/Mac, las partes con `for /f ... do set ...` deben adaptarse.

---

## Tabla de contenidos
- [Notas importantes](#notas-importantes)
- [0) EDA](#0-eda)
- [1) Preprocesado](#1-preprocesado)
- [2) Experimento 1](#2-experimento-1)
  - [Train](#train-exp1)
  - [Eval](#eval-exp1)
  - [XAI](#xai-exp1)
- [3) Experimento 2](#3-experimento-2)
  - [Coarse](#coarse-exp2)
  - [Fine](#fine-exp2)
- [4) Experimento 3](#4-experimento-3)
  - [Coarse](#coarse-exp3)
  - [Fine](#fine-exp3)
- [5) Experimento 4 – Pipeline final](#5-experimento-4--pipeline-final)
  - [0) Baseline multiclase (v4_full) — coarse → fine](#0-baseline-multiclase-v4_full--coarse--fine)
  - [1) Resolver BASE_BEST_PT](#1-resolver-base_best_pt)
  - [2) Crear dataset ROI-only (solo acera)](#2-crear-dataset-roi-only-solo-acera)
  - [3) Entrenar ROI-only (solo acera) — directo](#3-entrenar-roi-only-solo-acera--directo)
  - [4) Resolver ROI_BEST_PT](#4-resolver-roi_best_pt)
  - [4.1) (Recomendado) Decidir qué modelo de "acera" usar para crops](#41-recomendado-decidir-qué-modelo-de-acera-usar-para-crops)
  - [4.2) Elige el modelo de acera para crops (manual)](#42-elige-el-modelo-de-acera-para-crops-manual)
  - [5) Crear dataset de crops (ROI guiado por acera predicha)](#5-crear-dataset-de-crops-roi-guiado-por-acera-predicha)
  - [6) Entrenar detector final (solo obstáculo) sobre crops — coarse → fine](#6-entrenar-detector-final-solo-obstáculo-sobre-crops--coarse--fine)
  - [7) Resolver CROP_BEST_PT](#7-resolver-crop_best_pt)
  - [8) Hard negatives (sobre crops) + entrenamiento final (hn_fine)](#8-hard-negatives-sobre-crops--entrenamiento-final-hn_fine)
  - [9) Test final (cuando ya estés contento con VAL)](#9-test-final-cuando-ya-estés-contento-con-val)

---

## Notas importantes

- Este pipeline asume que:
  - Los datos están accesibles en las rutas esperadas por el proyecto.
  - Las configuraciones están en `configs/`.
  - Los scripts de `src/` generan sus outputs donde el proyecto los espera.
- Para ejecutar en **CMD / Anaconda Prompt**:
  - No uses `$(...)` ni `$VAR`.
  - Para capturar salida de un comando y guardarla en una variable se usa:
    - `for /f ... do set "VAR=..."`
  - Si copias comandos a un `.bat`, cambia `%i` por `%%i`.

---

## 0) EDA

    python src\0_eda\build_annotations_csv.py

---

## 1) Preprocesado

    python src\1_data\build_annotated_images.py
    python src\1_data\split_by_scene_clip_stratified.py

---

## 2) Experimento 1

### Train (Exp1)

    python src\2_experiments\exp1\0_train\make_yolo_dataset_exp1.py --config configs/exp1/base.yaml
    python src\2_experiments\exp1\0_train\run_exp1_train.py --config configs/exp1/base.yaml

### Eval (Exp1)

    python src\2_experiments\exp1\1_eval\extract_exp1_metrics.py --config configs/exp1/base.yaml
    python src\2_experiments\exp1\1_eval\build_metrics_last_focus_all.py --config configs/exp1/base.yaml --split val --target_class obstaculo --weights best

    python src\2_experiments\exp1\1_eval\aggregate_exp1_results.py --config configs/exp1/base.yaml --source focus
    python src\2_experiments\exp1\1_eval\aggregate_exp1_results.py --config configs/exp1/base.yaml --source all

    python src\2_experiments\exp1\1_eval\select_exp1_winner.py --config configs/exp1/base.yaml --source focus

    python src\2_experiments\exp1\1_eval\plot_exp1_results.py --config configs/exp1/base.yaml --source focus
    python src\2_experiments\exp1\1_eval\plot_exp1_quantitative.py --config configs/exp1/base.yaml --split val --n 12 --include_raw --with_gt

### XAI (Exp1)

    python src/2_experiments/exp1/2_xai/infer_exp1_predictions.py --config configs/exp1/base.yaml --split test --conf 0.25 --max_images 200 --target_class obstaculo
    python src/2_experiments/exp1/2_xai/xai_make_benchmark_list.py --config configs/exp1/base.yaml --split test --n 50 --seed 42 --use_stratified --pos_frac 0.6 --target_class obstaculo
    python src/2_experiments/exp1/2_xai/xai_drise_gt_yolo.py --config configs/exp1/base.yaml --target_class obstaculo --n_masks 600 --grid 16 --p 0.5 --conf 0.25 --seed 42
    python src/2_experiments/exp1/2_xai/xai_aggregate_gt.py --config configs/exp1/base.yaml
    python src/2_experiments/exp1/2_xai/xai_plot_results_gt.py --config configs/exp1/base.yaml
    python src/2_experiments/exp1/2_xai/xai_find_context_gain_cases.py --config configs/exp1/base.yaml --metric base_score --a v1_obstaculo --b v4_full --topk 30

**Casos donde el contexto ayuda**

    python src/2_experiments/exp1/2_xai/xai_make_panels_gt.py --config configs/exp1/base.yaml --list_csv cases_gain_base_score_v1_obstaculo_vs_v4_full.csv --paper_ab --a v1_obstaculo --b v4_full --seed 42 --max_images 30

**Casos donde el contexto no ayuda**

    python src/2_experiments/exp1/2_xai/xai_make_panels_gt.py --config configs/exp1/base.yaml --list_csv cases_drop_base_score_v1_obstaculo_vs_v4_full.csv --paper_ab --a v1_obstaculo --b v4_full --seed 42 --max_images 30

**Casos donde el contexto ayuda (diff heatmaps)**

    python src/2_experiments/exp1/2_xai/xai_diff_heatmaps_gt.py --config configs/exp1/base.yaml --list_csv cases_gain_base_score_v1_obstaculo_vs_v4_full.csv --seed 42 --a v1_obstaculo --b v4_full --max_images 30 --save_montage

**Casos donde el contexto no ayuda (diff heatmaps)**

    python src/2_experiments/exp1/2_xai/xai_diff_heatmaps_gt.py --config configs/exp1/base.yaml --list_csv cases_drop_base_score_v1_obstaculo_vs_v4_full.csv --seed 42 --a v1_obstaculo --b v4_full --max_images 30 --save_montage

---

## 3) Experimento 2

### Coarse (Exp2)

    python src/2_experiments/exp2/0_train/make_yolo_dataset_exp2.py --config configs/exp2/base.yaml --stage coarse
    python src/2_experiments/exp2/0_train/run_exp2_train.py --config configs/exp2/base.yaml --stage coarse

    python src/2_experiments/exp2/1_eval/build_metrics_best_focus_all.py --config configs/exp2/base.yaml --stage coarse --weights best --split val
    python src/2_experiments/exp2/1_eval/aggregate_exp2_results.py --config configs/exp2/base.yaml --stage coarse --source focus --weights best --split val
    python src/2_experiments/exp2/1_eval/select_exp2_winner.py --config configs/exp2/base.yaml --stage coarse --source focus --split val
    python src/2_experiments/exp2/1_eval/plot_exp2_results.py --config configs/exp2/base.yaml --stage coarse --source focus --split val --topk 4 --make_table --table_topk 8

### Fine (Exp2)

    python src/2_experiments/exp2/0_train/make_yolo_dataset_exp2.py --config configs/exp2/base.yaml --stage fine
    python src/2_experiments/exp2/0_train/run_exp2_train.py --config configs/exp2/base.yaml --stage fine

    python src/2_experiments/exp2/1_eval/build_metrics_best_focus_all.py --config configs/exp2/base.yaml --stage fine --weights best --split val
    python src/2_experiments/exp2/1_eval/aggregate_exp2_results.py --config configs/exp2/base.yaml --stage fine --source focus --weights best --split val
    python src/2_experiments/exp2/1_eval/select_exp2_winner.py --config configs/exp2/base.yaml --stage fine --source focus --split val
    python src/2_experiments/exp2/1_eval/plot_exp2_results.py --config configs/exp2/base.yaml --stage fine --source focus --split val --topk 4 --make_table --table_topk 8

---

## 4) Experimento 3

### Coarse (Exp3)

**Train**

    python src/2_experiments/exp3/0_train/make_yolo_dataset_exp3.py --config configs/exp3/base.yaml --stage coarse
    python src/2_experiments/exp3/0_train/run_exp3_train.py --config configs/exp3/base.yaml --stage coarse

**Eval threshold-free**

    python src/2_experiments/exp3/1_eval/build_metrics_best_focus_all.py --config configs/exp3/base.yaml --stage coarse --weights best --split val
    python src/2_experiments/exp3/1_eval/aggregate_exp3_results.py --config configs/exp3/base.yaml --stage coarse --source focus --weights best --split val
    python src/2_experiments/exp3/1_eval/select_exp3_winner.py --config configs/exp3/base.yaml --stage coarse --source focus --split val
    python src/2_experiments/exp3/1_eval/plot_exp3_results.py --config configs/exp3/base.yaml --stage coarse --source focus --split val --topk 4 --make_table --table_topk 8

**Eval operativo conf=0.25**

    python src/2_experiments/exp3/1_eval/build_metrics_best_focus_all.py --config configs/exp3/base.yaml --stage coarse --weights best --split val --conf 0.25
    python src/2_experiments/exp3/1_eval/aggregate_exp3_results.py --config configs/exp3/base.yaml --stage coarse --source focus --weights best --split val --conf 0.25
    python src/2_experiments/exp3/1_eval/select_exp3_winner.py --config configs/exp3/base.yaml --stage coarse --source focus --split val --conf 0.25
    python src/2_experiments/exp3/1_eval/plot_exp3_results.py --config configs/exp3/base.yaml --stage coarse --source focus --split val --conf 0.25 --topk 4 --make_table --table_topk 8

### Fine (Exp3)

**Train**

    python src/2_experiments/exp3/0_train/make_yolo_dataset_exp3.py --config configs/exp3/base.yaml --stage fine
    python src/2_experiments/exp3/0_train/run_exp3_train.py           --config configs/exp3/base.yaml --stage fine

**Eval threshold-free**

    python src/2_experiments/exp3/1_eval/build_metrics_best_focus_all.py --config configs/exp3/base.yaml --stage fine --weights best --split val
    python src/2_experiments/exp3/1_eval/aggregate_exp3_results.py       --config configs/exp3/base.yaml --stage fine --source focus --weights best --split val
    python src/2_experiments/exp3/1_eval/select_exp3_winner.py           --config configs/exp3/base.yaml --stage fine --source focus --split val
    python src/2_experiments/exp3/1_eval/plot_exp3_results.py            --config configs/exp3/base.yaml --stage fine --source focus --split val --topk 6 --make_table --table_topk 10

**Eval operativo conf=0.25**

    python src/2_experiments/exp3/1_eval/build_metrics_best_focus_all.py --config configs/exp3/base.yaml --stage fine --weights best --split val --conf 0.25
    python src/2_experiments/exp3/1_eval/aggregate_exp3_results.py       --config configs/exp3/base.yaml --stage fine --source focus --weights best --split val --conf 0.25
    python src/2_experiments/exp3/1_eval/select_exp3_winner.py           --config configs/exp3/base.yaml --stage fine --source focus --split val --conf 0.25
    python src/2_experiments/exp3/1_eval/plot_exp3_results.py            --config configs/exp3/base.yaml --stage fine --source focus --split val --conf 0.25 --topk 6 --make_table --table_topk 10

---

## 5) Experimento 4 — Pipeline final

Diseñado para **Anaconda Prompt / CMD (Windows)**.

IMPORTANTE:
- NO uses `$(...)` ni `$VAR` en CMD.
- En CMD se captura salida con: `for /f ... do set "VAR=..."`
- Si copias esto a un `.bat`, cambia `%i` por `%%i`.

---

### 0) Baseline multiclase (v4_full) — coarse → fine

**COARSE (grid)**

    python src/2_experiments/exp4/0_train/make_yolo_dataset_exp4.py --config configs/exp4/base.yaml --stage coarse --dataset_variant v5_full
    python src/2_experiments/exp4/0_train/run_exp4_train.py         --config configs/exp4/base.yaml --stage coarse

    python src/2_experiments/exp4/1_eval/build_metrics_best_focus_all.py --config configs/exp4/base.yaml --stage coarse --weights best --split val --force
    python src/2_experiments/exp4/1_eval/aggregate_exp4_results.py       --config configs/exp4/base.yaml --stage coarse --source focus --weights best --split val
    python src/2_experiments/exp4/1_eval/select_exp4_winner.py           --config configs/exp4/base.yaml --stage coarse --source focus --split val
    python src/2_experiments/exp4/1_eval/plot_exp4_results.py            --config configs/exp4/base.yaml --stage coarse --source focus --split val --topk 4 --make_table --table_topk 12

**FINE (top-k del coarse)**

    python src/2_experiments/exp4/0_train/make_yolo_dataset_exp4.py --config configs/exp4/base.yaml --stage fine --dataset_variant v5_full
    python src/2_experiments/exp4/0_train/run_exp4_train.py         --config configs/exp4/base.yaml --stage fine

    python src/2_experiments/exp4/1_eval/build_metrics_best_focus_all.py --config configs/exp4/base.yaml --stage fine --weights best --split val --force
    python src/2_experiments/exp4/1_eval/aggregate_exp4_results.py       --config configs/exp4/base.yaml --stage fine --source focus --weights best --split val
    python src/2_experiments/exp4/1_eval/select_exp4_winner.py           --config configs/exp4/base.yaml --stage fine --source focus --split val
    python src/2_experiments/exp4/1_eval/plot_exp4_results.py            --config configs/exp4/base.yaml --stage fine --source focus --split val --topk 6 --make_table --table_topk 10

**(opcional) sweep operating point en val para baseline**

    python src/2_experiments/exp4/1_eval/sweep_operating_point.py        --config configs/exp4/base.yaml --stage fine --split val --weights best

---

### 1) Resolver BASE_BEST_PT (ganador del baseline)

    for /f "usebackq delims=" %i in (`python src/2_experiments/exp4/0_train/resolve_best_pt.py --config configs/exp4/base.yaml --stage fine --source focus --split val --weights best`) do set "BASE_BEST_PT=%i"
    echo [INFO] BASE_BEST_PT=%BASE_BEST_PT%

---

### 2) Crear dataset ROI-only (SOLO ACERA)

    python src/2_experiments/exp4/0_train/make_single_class_dataset.py --config configs/exp4/base.yaml --base_dataset_variant v5_full --target_class acera --out_variant v5_acera_only

---

### 3) Entrenar ROI-only (SOLO ACERA) — DIRECTO (SIN GRID)

    python src/2_experiments/exp4/0_train/make_yolo_dataset_exp4.py --config configs/exp4/base.yaml --stage roi_train --dataset_variant v5_acera_only --model "%BASE_BEST_PT%"
    python src/2_experiments/exp4/0_train/run_exp4_train.py         --config configs/exp4/base.yaml --stage roi_train

    python src/2_experiments/exp4/1_eval/build_metrics_best_focus_all.py --config configs/exp4/base.yaml --stage roi_train --weights best --split val --target_class acera --force
    python src/2_experiments/exp4/1_eval/aggregate_exp4_results.py       --config configs/exp4/base.yaml --stage roi_train --source focus --weights best --split val
    python src/2_experiments/exp4/1_eval/select_exp4_winner.py           --config configs/exp4/base.yaml --stage roi_train --source focus --split val
    python src/2_experiments/exp4/1_eval/plot_exp4_results.py            --config configs/exp4/base.yaml --stage roi_train --source focus --split val --topk 6 --make_table --table_topk 10

---

### 4) Resolver ROI_BEST_PT (ganador ROI-only)

    for /f "usebackq delims=" %i in (`python src/2_experiments/exp4/0_train/resolve_best_pt.py --config configs/exp4/base.yaml --stage roi_train --source focus --split val --weights best`) do set "ROI_BEST_PT=%i"
    echo [INFO] ROI_BEST_PT=%ROI_BEST_PT%

---

### 4.1) (RECOMENDADO) Decidir qué modelo de "ACERA" usar para CROPS

    python src/2_experiments/exp4/1_eval/eval_single_model_focus.py --config configs/exp4/base.yaml --dataset_yaml data/meta/exp1/datasets/v5_full/dataset.yaml --weights "%BASE_BEST_PT%" --split val --target_class acera --imgsz 960 --device 0 --conf 0.25 --iou_match 0.5
    python src/2_experiments/exp4/1_eval/eval_single_model_focus.py --config configs/exp4/base.yaml --dataset_yaml data/meta/exp1/datasets/v5_full/dataset.yaml --weights "%ROI_BEST_PT%"  --split val --target_class acera --imgsz 960 --device 0 --conf 0.25 --iou_match 0.5

---

### 4.2) Elige el modelo de ACERA para CROPS (manual)

    set "SIDEWALK_WEIGHTS=%ROI_BEST_PT%"
    rem set "SIDEWALK_WEIGHTS=%BASE_BEST_PT%"
    echo [INFO] SIDEWALK_WEIGHTS=%SIDEWALK_WEIGHTS%

---

### 5) Crear dataset de CROPS (ROI guiado por ACERA predicha)

    python src/2_experiments/exp4/0_train/make_sidewalk_crop_dataset.py --config configs/exp4/base.yaml --base_dataset_variant v5_full --weights "%SIDEWALK_WEIGHTS%" --out_variant v5_sidewalkcrop_obstaculo

---

### 6) Entrenar detector final (solo obstáculo) sobre CROPS — COARSE → FINE

**crop_coarse**

    python src/2_experiments/exp4/0_train/make_yolo_dataset_exp4.py --config configs/exp4/base.yaml --stage crop_coarse --dataset_variant v5_sidewalkcrop_obstaculo --model "%BASE_BEST_PT%"
    python src/2_experiments/exp4/0_train/run_exp4_train.py         --config configs/exp4/base.yaml --stage crop_coarse

    python src/2_experiments/exp4/1_eval/build_metrics_best_focus_all.py --config configs/exp4/base.yaml --stage crop_coarse --weights best --split val --force
    python src/2_experiments/exp4/1_eval/aggregate_exp4_results.py       --config configs/exp4/base.yaml --stage crop_coarse --source focus --weights best --split val
    python src/2_experiments/exp4/1_eval/select_exp4_winner.py           --config configs/exp4/base.yaml --stage crop_coarse --source focus --split val
    python src/2_experiments/exp4/1_eval/plot_exp4_results.py            --config configs/exp4/base.yaml --stage crop_coarse --source focus --split val --topk 8 --make_table --table_topk 12

**crop_fine**

    python src/2_experiments/exp4/0_train/make_yolo_dataset_exp4.py --config configs/exp4/base.yaml --stage crop_fine --dataset_variant v4_sidewalkcrop_obstaculo --model "%BASE_BEST_PT%"
    python src/2_experiments/exp4/0_train/run_exp4_train.py         --config configs/exp4/base.yaml --stage crop_fine

    python src/2_experiments/exp4/1_eval/build_metrics_best_focus_all.py --config configs/exp4/base.yaml --stage crop_fine --weights best --split val --force
    python src/2_experiments/exp4/1_eval/aggregate_exp4_results.py       --config configs/exp4/base.yaml --stage crop_fine --source focus --weights best --split val
    python src/2_experiments/exp4/1_eval/select_exp4_winner.py           --config configs/exp4/base.yaml --stage crop_fine --source focus --split val
    python src/2_experiments/exp4/1_eval/plot_exp4_results.py            --config configs/exp4/base.yaml --stage crop_fine --source focus --split val --topk 6 --make_table --table_topk 10

**sweep operating point (VAL) para buscar conf/iou óptimos (objetivo 0.70/0.70)**

    python src/2_experiments/exp4/1_eval/sweep_operating_point.py        --config configs/exp4/base.yaml --stage crop_fine --split val --weights best

---

### 7) Resolver CROP_BEST_PT (ganador crop_fine)

    for /f "usebackq delims=" %i in (`python src/2_experiments/exp4/0_train/resolve_best_pt.py --config configs/exp4/base.yaml --stage crop_coarse --source focus --split val --weights best`) do set "CROP_BEST_PT=%i"
    echo [INFO] CROP_BEST_PT=%CROP_BEST_PT%

---

### 8) Hard negatives (sobre CROPS) + entrenamiento final (hn_fine)

    python src/2_experiments/exp4/0_train/mine_hard_negatives.py --dataset_yaml data/meta/exp4/datasets/v5_sidewalkcrop_obstaculo/dataset.yaml --weights "%CROP_BEST_PT%" --split train --conf 0.25 --iou 0.6 --device 0 --out_dir data/meta/exp4/datasets/v5_sidewalkcrop_obstaculo_hardneg --repeat 2

    python src/2_experiments/exp4/0_train/make_yolo_dataset_exp4.py --config configs/exp4/base.yaml --stage hn_fine --dataset_variant v5_sidewalkcrop_obstaculo_hardneg --model "%CROP_BEST_PT%"
    python src/2_experiments/exp4/0_train/run_exp4_train.py         --config configs/exp4/base.yaml --stage hn_fine

    python src/2_experiments/exp4/1_eval/build_metrics_best_focus_all.py --config configs/exp4/base.yaml --stage hn_fine --weights best --split val --force
    python src/2_experiments/exp4/1_eval/aggregate_exp4_results.py       --config configs/exp4/base.yaml --stage hn_fine --source focus --weights best --split val
    python src/2_experiments/exp4/1_eval/select_exp4_winner.py           --config configs/exp4/base.yaml --stage hn_fine --source focus --split val
    python src/2_experiments/exp4/1_eval/plot_exp4_results.py            --config configs/exp4/base.yaml --stage hn_fine --source focus --split val --topk 6 --make_table --table_topk 10

    python src/2_experiments/exp4/1_eval/sweep_operating_point.py        --config configs/exp4/base.yaml --stage hn_fine --split val --weights best

---

### 9) Test final (cuando ya estés contento con VAL)

    python src/2_experiments/exp4/1_eval/build_metrics_best_focus_all.py --config configs/exp4/base.yaml --stage hn_fine --weights best --split test --force
    python src/2_experiments/exp4/1_eval/sweep_operating_point.py        --config configs/exp4/base.yaml --stage coarse --split test --weights best

---