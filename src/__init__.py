"""
Paquete de lógica del proyecto de inspección visual.

`config`   — rutas del proyecto y configuración del pipeline.
`inspection` — lógica de decisión compartida entre el pipeline y la aplicación web.
"""

from .config import (
    CLASS_LABELS,
    CLASS_MAP,
    CLASS_NAMES,
    CONFIG,
    DATA_DIR,
    DOCS_DIR,
    EXPORTS_DIR,
    FIGURES_DIR,
    INSPECTIONS_DIR,
    MODELS_DIR,
    PRETRAINED_DIR,
    REPORT_DOCX,
    REPORT_TXT,
    RESULTS_DIR,
    ROOT,
    RUNS_DIR,
    TRAINED_DIR,
    create_directories,
    pretrained_weights,
    trained_model,
)

__all__ = [
    "CLASS_LABELS",
    "CLASS_MAP",
    "CLASS_NAMES",
    "CONFIG",
    "DATA_DIR",
    "DOCS_DIR",
    "EXPORTS_DIR",
    "FIGURES_DIR",
    "INSPECTIONS_DIR",
    "MODELS_DIR",
    "PRETRAINED_DIR",
    "REPORT_DOCX",
    "REPORT_TXT",
    "RESULTS_DIR",
    "ROOT",
    "RUNS_DIR",
    "TRAINED_DIR",
    "create_directories",
    "pretrained_weights",
    "trained_model",
]
