"""
================================================================================
CONFIGURACIÓN DEL PROYECTO
================================================================================

Define la configuración global y las rutas del proyecto.

Todas las rutas se resuelven a partir de la raíz del repositorio, calculada de
forma relativa a este archivo. De este modo los scripts funcionan con
independencia del directorio desde el que se invoquen, algo que las rutas
relativas al directorio de trabajo (`./results`) no garantizaban.
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import torch

# ==============================================================================
# RUTAS DEL PROYECTO
# ==============================================================================
# config.py -> src/ -> raíz del proyecto
ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"

# Datos
MVTEC_DIR = DATA_DIR / "mvtec_ad"
YOLO_DATASET_DIR = DATA_DIR / "yolo_mvtec"
DATA_YAML = YOLO_DATASET_DIR / "data.yaml"

# Modelos
PRETRAINED_DIR = MODELS_DIR / "pretrained"
TRAINED_DIR = MODELS_DIR / "trained"
EXPORTS_DIR = MODELS_DIR / "exports"

# Resultados
FIGURES_DIR = RESULTS_DIR / "figuras"
INSPECTIONS_DIR = RESULTS_DIR / "inspections"
RUN_NAME = "yolov8_inspection"

# Documentos
REPORT_DOCX = DOCS_DIR / "proyecto_inspeccion_visual.docx"
REPORT_TXT = DOCS_DIR / "proyecto_inspeccion_visual.txt"


# ==============================================================================
# CONFIGURACIÓN DEL PIPELINE
# ==============================================================================
CONFIG = {
    "category": "bottle",           # Categoría del conjunto MVTec AD
    "img_size": 640,                # Tamaño de entrada de YOLOv8
    "batch_size": 16,
    "epochs": 100,
    "device": "0" if torch.cuda.is_available() else "cpu",
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "model_size": "yolov8s.pt",     # YOLOv8 small (balance precisión/velocidad)
    "val_ratio": 0.25,              # Proporción de validación dentro de la partición de prueba
    "seed": 42,                     # Semilla del particionado y del muestreo
    "run_name": RUN_NAME,           # Carpeta de la ejecución de entrenamiento
}

# Nombres de las clases, en el orden de índices que usa YOLO
CLASS_NAMES = ["good", "broken_large", "broken_small", "contamination"]
CLASS_MAP = {nombre: i for i, nombre in enumerate(CLASS_NAMES)}

# Etiquetas legibles para los informes y la interfaz
CLASS_LABELS = {
    "good": "Normal",
    "broken_large": "Grieta grande",
    "broken_small": "Grieta pequeña",
    "contamination": "Contaminación",
}


def create_directories() -> None:
    """Crea la estructura de directorios de salida si no existe."""
    for carpeta in (RESULTS_DIR, MODELS_DIR, PRETRAINED_DIR, TRAINED_DIR,
                    EXPORTS_DIR, FIGURES_DIR, INSPECTIONS_DIR):
        carpeta.mkdir(parents=True, exist_ok=True)


def pretrained_weights() -> Path:
    """
    Devuelve la ruta de los pesos preentrenados de COCO.

    Si no existen en `models/pretrained/`, se conserva el nombre simple para que
    Ultralytics lo descargue automáticamente en la raíz. Esto mantiene el
    proyecto funcional en una instalación limpia.
    """
    candidato = PRETRAINED_DIR / CONFIG["model_size"]
    return candidato if candidato.exists() else Path(CONFIG["model_size"])


def trained_model() -> Path:
    """Ruta del modelo entrenado listo para inferencia."""
    return TRAINED_DIR / "best_yolov8_inspection.pt"
