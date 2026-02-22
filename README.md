# TFM – Análisis de accesibilidad urbana para personas con movilidad reducida mediante Visión Artificial

Este repositorio contiene el código y los experimentos del Trabajo Fin de Máster orientado al **análisis de accesibilidad urbana** para personas con **movilidad reducida** mediante **visión artificial**.  
El objetivo principal es entrenar y evaluar un modelo capaz de **detectar obstáculos** en entornos urbanos y estudiar distintas **hipótesis experimentales** a lo largo del trabajo.

## Contenido del repositorio
- `src/` – Código del pipeline (EDA, preprocesado, entrenamiento, evaluación, XAI).
- `configs/` – Configuraciones YAML por experimento (exp1–exp4).
- `docs/PIPELINE.md` – Guía detallada con el **orden exacto de ejecución** de todos los pasos y experimentos.

## Reproducibilidad (orden de ejecución)
Para ejecutar el proyecto en el orden correcto (EDA → preprocesado → experimentos → evaluación), consulta:

- **`docs/PIPELINE.md`**

## Requisitos
- Python **3.10+** (recomendado)
- GPU NVIDIA + CUDA (recomendado para entrenamiento). El proyecto también puede ejecutarse en CPU, pero será más lento.

### Instalación rápida

pip install -r requirements.txt
