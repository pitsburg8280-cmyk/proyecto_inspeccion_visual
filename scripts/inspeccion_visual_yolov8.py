
"""
================================================================================
PIPELINE DE INSPECCIÓN VISUAL — YOLOv8 + MVTec AD
================================================================================
Ejecuta las seis fases del proyecto:

    1. Preparación del conjunto de datos (inventario y verificación)
    2. Exploración y visualización (EDA)
    3. Conversión a formato YOLO y entrenamiento
    4. Evaluación: métricas, matriz de confusión y latencia
    5. Inspección por lotes
    6. Exportación del modelo (ONNX y TorchScript)

USO:
    python scripts/inspeccion_visual_yolov8.py            # pipeline completo
    python scripts/inspeccion_visual_yolov8.py evaluar    # sin reentrenar

La configuración y las rutas se definen en `src/config.py`, y la lógica de
decisión compartida con la aplicación web en `src/inspection.py`.
================================================================================
"""

import os
import sys
import cv2
import time
import json
import random
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Permite importar el paquete `src` al ejecutar el script desde cualquier lugar.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CLASS_LABELS,
    CLASS_MAP,
    CLASS_NAMES,
    CONFIG,
    DATA_YAML,
    DOCS_DIR,
    EXPORTS_DIR,
    FIGURES_DIR,
    INSPECTIONS_DIR,
    MODELS_DIR,
    MVTEC_DIR,
    PRETRAINED_DIR,
    REPORT_DOCX,
    REPORT_TXT,
    RESULTS_DIR,
    ROOT,
    RUNS_DIR,
    TRAINED_DIR,
    YOLO_DATASET_DIR,
    create_directories,
    pretrained_weights,
    trained_model,
)
from src.inspection import analizar_predicciones, decidir, inspeccionar  # noqa: E402

create_directories()

print(f"[INFO] Raíz del proyecto: {ROOT}")
print(f"[INFO] Dispositivo: {CONFIG['device']}")
print(f"[INFO] CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# FASE 1: PREPARACIÓN DEL DATASET MVTec AD
# ==============================================================================

def prepare_mvtec_dataset():
    """
    Prepara el dataset MVTec AD para entrenamiento con YOLOv8.
    Convierte las anotaciones al formato YOLO (txt con bounding boxes).
    """
    print("\n" + "="*70)
    print("FASE 1: PREPARACIÓN DEL DATASET MVTec AD")
    print("="*70)

    # El dataset MVTec AD tiene la siguiente estructura:
    # mvtec_ad/bottle/
    #   ├── train/
    #   │   └── good/           # Solo imágenes normales
    #   └── test/
    #       ├── good/           # Imágenes normales
    #       ├── broken_large/   # Defecto tipo 1
    #       ├── broken_small/   # Defecto tipo 2
    #       └── contamination/  # Defecto tipo 3
    #   └── ground_truth/       # Máscaras de segmentación

    dataset_path = MVTEC_DIR / CONFIG["category"]

    if not dataset_path.exists():
        print(f"[ERROR] Dataset no encontrado en: {dataset_path}")
        print("[INFO] Genere la réplica sintética con:")
        print("       python generar_dataset_sintetico.py")
        print("[INFO] O descargue el MVTec AD oficial desde:")
        print("       https://www.mvtec.com/company/research/datasets/mvtec-ad")
        return False

    # Inventario de todas las clases presentes en cada partición.
    def inventario(split):
        raiz = dataset_path / split
        if not raiz.exists():
            return {}
        return {
            d.name: len(list(d.glob("*.png")))
            for d in sorted(raiz.iterdir())
            if d.is_dir()
        }

    train_counts = inventario("train")
    test_counts = inventario("test")

    print(f"[INFO] Categoría: {CONFIG['category']}")
    for split, conteo in (("entrenamiento", train_counts), ("prueba", test_counts)):
        total = sum(conteo.values())
        print(f"[INFO] Partición de {split}: {total} imágenes -> {conteo}")

    # Un detector supervisado necesita ejemplos positivos de cada defecto.
    clases_defecto = {k for k in test_counts if k != "good"}
    faltantes = sorted(clases_defecto - set(train_counts))
    if faltantes:
        print(f"[WARN] La partición de entrenamiento no contiene ejemplos de: {faltantes}")
        print("[WARN] El modelo no podrá aprender esas clases. Ejecute:")
        print("       python generar_dataset_sintetico.py")

    # Verificar la correspondencia entre imágenes defectuosas y sus máscaras.
    sin_mascara = 0
    for clase, n in train_counts.items():
        if clase == "good":
            continue
        gt = dataset_path / "ground_truth" / clase
        for img in (dataset_path / "train" / clase).glob("*.png"):
            if not (gt / f"{img.stem}_mask.png").exists():
                sin_mascara += 1
    if sin_mascara:
        print(f"[WARN] {sin_mascara} imágenes de entrenamiento sin máscara asociada;")
        print("       se usarán cuadros aproximados en su lugar.")

    # Crear resumen del dataset
    summary = {
        "categoria": CONFIG["category"],
        "train": train_counts,
        "test": test_counts,
        "train_normal": train_counts.get("good", 0),
        "total_entrenamiento": sum(train_counts.values()),
        "total_prueba": sum(test_counts.values()),
        "total_defectos": sum(v for k, v in test_counts.items() if k != "good"),
        "tipos_defecto": sorted(clases_defecto),
        "entrenamiento_supervisado": not faltantes,
    }

    with open(RESULTS_DIR / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Resumen guardado en: {RESULTS_DIR / 'dataset_summary.json'}")
    return True


# ==============================================================================
# FASE 2: EXPLORACIÓN Y VISUALIZACIÓN DE DATOS
# ==============================================================================

def explore_dataset():
    """
    Realiza análisis exploratorio de datos (EDA) visualizando muestras.
    """
    print("\n" + "="*70)
    print("FASE 2: EXPLORACIÓN Y VISUALIZACIÓN DE DATOS")
    print("="*70)

    dataset_path = MVTEC_DIR / CONFIG["category"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Muestras del Dataset MVTec AD - Categoría: {CONFIG['category'].upper()}", 
                 fontsize=14, fontweight='bold')

    # Muestras de entrenamiento (normales)
    train_dir = dataset_path / "train" / "good"
    train_samples = sorted(list(train_dir.glob("*.png")))[:4]

    for idx, img_path in enumerate(train_samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, idx].imshow(img)
        axes[0, idx].set_title("Entrenamiento: Normal", color='green', fontweight='bold')
        axes[0, idx].axis('off')

    # Muestras de prueba (con defectos)
    test_dir = dataset_path / "test"
    defect_types = [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]

    for idx, defect_dir in enumerate(defect_types[:4]):
        samples = sorted(list(defect_dir.glob("*.png")))
        if samples:
            img = cv2.imread(str(samples[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[1, idx].imshow(img)
            axes[1, idx].set_title(f"Prueba: {defect_dir.name}", color='red', fontweight='bold')
            axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dataset_exploration.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Visualización guardada en: {RESULTS_DIR / 'dataset_exploration.png'}")

    # Estadísticas de píxeles
    print("\n[INFO] Estadísticas de píxeles (muestra de entrenamiento):")
    pixel_values = []
    for img_path in train_samples[:20]:
        img = cv2.imread(str(img_path))
        pixel_values.extend(img.flatten())

    print(f"       Media: {np.mean(pixel_values):.2f}")
    print(f"       Desviación estándar: {np.std(pixel_values):.2f}")
    print(f"       Mín: {np.min(pixel_values)}, Máx: {np.max(pixel_values)}")


# ==============================================================================
# FASE 3: ENTRENAMIENTO DEL MODELO YOLOv8
# ==============================================================================

def convert_to_yolo_format():
    """
    Convierte el dataset MVTec AD al formato YOLO.
    Para defectos sin bounding boxes explícitos, genera boxes a partir de máscaras.
    """
    print("\n" + "="*70)
    print("FASE 3.1: CONVERSIÓN A FORMATO YOLO")
    print("="*70)

    yolo_dataset = YOLO_DATASET_DIR
    yolo_dataset.mkdir(parents=True, exist_ok=True)

    # Crear estructura YOLO
    for split in ["train", "val", "test"]:
        (yolo_dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    dataset_path = MVTEC_DIR / CONFIG["category"]

    # Clases: 0 = good (normal), 1 = broken_large, 2 = broken_small, 3 = contamination
    class_map = CLASS_MAP
    class_map_inv = {str(v): k for k, v in class_map.items()}

    def process_split(source_split, target_split, is_train=False):
        """Procesa imágenes de un split y genera labels YOLO."""
        root = dataset_path / source_split
        # Todas las clases presentes en el split (good + tipos de defecto).
        # Un detector supervisado requiere ejemplos positivos de cada defecto:
        # entrenar solo con imágenes normales impediría aprender las clases 1-3.
        source_dirs = [(d, d.name) for d in sorted(root.iterdir()) if d.is_dir()]

        img_idx = 0
        for source_dir, class_name in source_dirs:
            if class_name not in class_map:
                continue

            for img_path in sorted(source_dir.glob("*.png")):
                # Copiar imagen
                target_img = yolo_dataset / "images" / target_split / f"{img_idx:04d}.png"
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                cv2.imwrite(str(target_img), img)

                # Generar label
                label_path = yolo_dataset / "labels" / target_split / f"{img_idx:04d}.txt"
                class_id = class_map[class_name]

                if class_name == "good":
                    # Imagen normal: sin defectos. Se deja el archivo de etiquetas
                    # vacío para que YOLO la trate como fondo (negativo), en lugar
                    # de asignarle un cuadro de clase 0 que cubra toda la imagen;
                    # esto evita que el modelo aprenda a etiquetar el fondo entero.
                    open(label_path, 'w').close()
                else:
                    # Intentar cargar máscara para generar bounding box
                    mask_path = dataset_path / "ground_truth" / class_name / img_path.name.replace(".png", "_mask.png")

                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        coords = cv2.findNonZero(mask)
                        if coords is not None:
                            x, y, bw, bh = cv2.boundingRect(coords)
                            # Normalizar a formato YOLO (x_center, y_center, width, height)
                            xc = (x + bw/2) / w
                            yc = (y + bh/2) / h
                            nw = bw / w
                            nh = bh / h
                            with open(label_path, 'w') as f:
                                f.write(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                    else:
                        # Sin máscara: usar box centrado pequeño como aproximación
                        with open(label_path, 'w') as f:
                            f.write(f"{class_id} 0.5 0.5 0.3 0.3\n")

                img_idx += 1

        return img_idx

    # Limpiar particiones derivadas para que el pipeline sea idempotente:
    # sin esto, re-ejecutarlo acumula imágenes y desplaza las proporciones.
    for particion in ("val", "test"):
        for sub in ("images", "labels"):
            carpeta = yolo_dataset / sub / particion
            carpeta.mkdir(parents=True, exist_ok=True)
            for archivo in carpeta.iterdir():
                if archivo.is_file():
                    archivo.unlink()

    # Procesar splits
    n_train = process_split("train", "train", is_train=True)
    n_test = process_split("test", "test", is_train=False)

    # Particionado estratificado del split de prueba en validación y prueba.
    # Se reparte cada clase de forma proporcional para que ambas particiones
    # contengan ejemplos normales y de los tres tipos de defecto.
    import random

    test_images = sorted((yolo_dataset / "images" / "test").glob("*.png"))
    rng = random.Random(CONFIG["seed"])

    por_clase = {}
    for img_path in test_images:
        label = yolo_dataset / "labels" / "test" / f"{img_path.stem}.txt"
        contenido = label.read_text().strip()
        clase = contenido.split()[0] if contenido else "good"
        por_clase.setdefault(clase, []).append(img_path)

    n_val = 0
    for clase, rutas in sorted(por_clase.items()):
        rng.shuffle(rutas)
        n_clase_val = max(1, round(len(rutas) * CONFIG["val_ratio"]))
        for img_path in rutas[:n_clase_val]:
            origen_label = yolo_dataset / "labels" / "test" / f"{img_path.stem}.txt"
            img_path.rename(yolo_dataset / "images" / "val" / img_path.name)
            if origen_label.exists():
                origen_label.rename(yolo_dataset / "labels" / "val" / origen_label.name)
            n_val += 1

    n_test_final = len(list((yolo_dataset / "images" / "test").glob("*.png")))
    print(f"[INFO] Particionado estratificado -> validación: {n_val}, prueba: {n_test_final}")

    # Registrar la composición real de cada partición para el informe
    composicion = {}
    for particion in ("train", "val", "test"):
        archivos = sorted((yolo_dataset / "labels" / particion).glob("*.txt"))
        conteo = {}
        for label in archivos:
            contenido = label.read_text().strip()
            clave = class_map_inv.get(contenido.split()[0], "good") if contenido else "good"
            conteo[clave] = conteo.get(clave, 0) + 1
        composicion[particion] = {"archivos": len(archivos), "instancias": conteo}

    with open(RESULTS_DIR / "split_composition.json", 'w') as f:
        json.dump(composicion, f, indent=4)

    # Crear archivo data.yaml
    data_yaml = f"""
path: {yolo_dataset.absolute()}
train: images/train
val: images/val
test: images/test

nc: 4
names: ['good', 'broken_large', 'broken_small', 'contamination']
"""
    with open(yolo_dataset / "data.yaml", 'w') as f:
        f.write(data_yaml.strip())

    print(f"[INFO] Dataset YOLO creado: {yolo_dataset}")
    print(f"[INFO] Imágenes de entrenamiento: {n_train}")
    print(f"[INFO] Imágenes de validación: {n_val}")
    print(f"[INFO] Imágenes de prueba: {n_test_final}")
    print(f"[INFO] Archivo data.yaml creado")

    return str(yolo_dataset / "data.yaml")


def train_model(data_yaml_path):
    """
    Entrena el modelo YOLOv8 para detección de defectos.
    """
    print("\n" + "="*70)
    print("FASE 3.2: ENTRENAMIENTO DEL MODELO YOLOv8")
    print("="*70)

    # Cargar modelo pre-entrenado
    model = YOLO(str(pretrained_weights()))

    print(f"[INFO] Modelo base: {CONFIG['model_size']}")
    print(f"[INFO] Tamaño de imagen: {CONFIG['img_size']}")
    print(f"[INFO] Batch size: {CONFIG['batch_size']}")
    print(f"[INFO] Épocas: {CONFIG['epochs']}")
    print(f"[INFO] Dispositivo: {CONFIG['device']}")

    # Entrenamiento
    start_time = time.time()

    results = model.train(
        data=data_yaml_path,
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["img_size"],
        batch=CONFIG["batch_size"],
        device=CONFIG["device"],
        project=str(RUNS_DIR / "detect"),
        name=CONFIG.get("run_name", "yolov8_inspection"),
        exist_ok=True,
        patience=10,                    # Early stopping
        save=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        augment=True,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    training_time = time.time() - start_time

    print(f"\n[INFO] Entrenamiento completado en: {training_time:.2f} segundos")
    print(f"[INFO] Mejor modelo guardado en: {best_model_src}")

    # Guardar métricas de entrenamiento
    train_metrics = {
        "training_time_seconds": training_time,
        "epochs": CONFIG["epochs"],
        "best_map50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
        "best_map50_95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
        "final_precision": float(results.results_dict.get('metrics/precision(B)', 0)),
        "final_recall": float(results.results_dict.get('metrics/recall(B)', 0)),
    }

    with open(RESULTS_DIR / "training_metrics.json", 'w') as f:
        json.dump(train_metrics, f, indent=4)

    # Copiar mejor modelo
    best_model_src = Path(getattr(results, "save_dir", "")) / "weights" / "best.pt"
    best_model_dst = TRAINED_DIR / "best_yolov8_inspection.pt"

    if best_model_src.exists():
        import shutil
        best_model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(best_model_src), str(best_model_dst))
        print(f"[INFO] Modelo copiado a: {best_model_dst}")
    else:
        print(f"[WARN] No se encontró el modelo entrenado en: {best_model_src}")
        raise FileNotFoundError(f"Modelo no encontrado: {best_model_src}")

    return str(best_model_dst)


# ==============================================================================
# FASE 4: EVALUACIÓN Y MÉTRICAS
# ==============================================================================

def evaluate_model(model_path, data_yaml_path):
    """
    Evalúa el modelo entrenado y calcula métricas detalladas.
    """
    print("\n" + "="*70)
    print("FASE 4: EVALUACIÓN Y MÉTRICAS DEL MODELO")
    print("="*70)

    model = YOLO(model_path)

    print("\n[INFO] Ejecutando validación oficial de YOLOv8...")
    val_results = model.val(
        data=data_yaml_path,
        split='test',
        imgsz=CONFIG["img_size"],
        batch=CONFIG["batch_size"],
        device=CONFIG["device"],
        project=str(RESULTS_DIR),
        name="validation",
        exist_ok=True,
        conf=CONFIG["confidence_threshold"],
        iou=CONFIG["iou_threshold"],
        save_json=True,
        save_hybrid=True,
    )

    # Extraer métricas
    metrics = {
        "mAP50": float(val_results.box.map50),
        "mAP50_95": float(val_results.box.map),
        "mAP": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "mean_precision": float(val_results.box.p.mean()),
        "mean_recall": float(val_results.box.r.mean()),
        "mean_f1": float(val_results.box.f1[np.isfinite(val_results.box.f1)].mean())
            if np.isfinite(val_results.box.f1).any() else 0.0,
    }

    # Métricas desagregadas por clase (AP@0.5 y AP@0.5:0.95)
    nombres_clase = [val_results.names[i] for i in sorted(val_results.names)]
    por_clase = []
    for pos, clase_idx in enumerate(val_results.box.ap_class_index):
        por_clase.append({
            "clase": nombres_clase[int(clase_idx)],
            "precision": float(val_results.box.p[pos]),
            "recall": float(val_results.box.r[pos]),
            "f1": float(val_results.box.f1[pos]),
            "ap50": float(val_results.box.ap50[pos]),
            "ap50_95": float(val_results.box.ap[pos]),
        })
    metrics["por_clase"] = por_clase

    # Matriz de confusión numérica. Se calcula con emparejamiento explícito por
    # IoU para que cada fila sume el total de instancias reales de esa clase,
    # semántica que la matriz interna de Ultralytics no garantiza en esta versión.
    print("\n[INFO] Calculando matriz de confusión...")
    cm = compute_confusion_matrix(
        model,
        Path(str(YOLO_DATASET_DIR / "images/test")),
        Path(str(YOLO_DATASET_DIR / "labels/test")),
        nombres_clase,
        conf=CONFIG["confidence_threshold"],
        iou=CONFIG["iou_threshold"],
        device=CONFIG["device"],
    )
    with open(RESULTS_DIR / "confusion_matrix.json", 'w') as f:
        json.dump({
            "clases": nombres_clase + ["fondo"],
            "matriz": [[int(v) for v in fila] for fila in cm],
        }, f, indent=4)
    print("[INFO] Matriz de confusión (filas = real, columnas = predicho):")
    print(f"       clases: {nombres_clase + ['fondo']}")
    for i, fila in enumerate(cm.tolist()):
        etiqueta = nombres_clase[i] if i < len(nombres_clase) else "fondo"
        print(f"       {etiqueta:<16} {fila}")

    print(f"\n{'='*50}")
    print("MÉTRICAS DE DESEMPAÑO")
    print(f"{'='*50}")
    print(f"mAP@0.5:      {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.2f}%)")
    print(f"mAP@0.5:0.95: {metrics['mAP50_95']:.4f} ({metrics['mAP50_95']*100:.2f}%)")
    print(f"mAP:          {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)")
    print(f"Precisión:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:     {metrics['mean_f1']:.4f} ({metrics['mean_f1']*100:.2f}%)")
    print(f"{'='*50}")

    # Guardar métricas
    with open(RESULTS_DIR / "evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    # Evaluación de latencia (inferencia)
    print("\n[INFO] Midiendo latencia de inferencia...")

    # Cargar imagen de prueba
    test_img_path = list(Path(str(YOLO_DATASET_DIR / "images/test")).glob("*.png"))[0]
    test_img = cv2.imread(str(test_img_path))

    latencies = []
    n_runs = 100

    # Calentamiento. Además de las tandas iniciales habituales, se sincroniza la
    # GPU antes de medir: sin ello, la inicialización diferida de CUDA y de los
    # kernels de la primera inferencia inflarían la latencia media de forma
    # artificial y haría que las mediciones no fueran reproducibles.
    for _ in range(10):
        model.predict(test_img, verbose=False, device=CONFIG["device"])

    def _sincronizar():
        """Bloquea hasta que la GPU termina el trabajo encolado.

        Las operaciones de CUDA son asíncronas: sin esta barrera, el cronómetro
        mediría únicamente el tiempo de encolado de los kernels en lugar del
        tiempo real de cómputo.
        """
        if CONFIG["device"] != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

    _sincronizar()

    # Medición
    for _ in range(n_runs):
        _sincronizar()
        start = time.perf_counter()
        model.predict(test_img, verbose=False, device=CONFIG["device"])
        _sincronizar()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    # Se conservan las mediciones individuales para graficar la distribución real
    latency_metrics = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": float(1000 / np.mean(latencies)),
        "muestras_ms": [float(v) for v in latencies],
    }

    print(f"\n{'='*50}")
    print("MÉTRICAS DE LATENCIA (100 ejecuciones)")
    print(f"{'='*50}")
    print(f"Latencia media:   {latency_metrics['mean_ms']:.2f} ms")
    print(f"Desv. estándar:   {latency_metrics['std_ms']:.2f} ms")
    print(f"Latencia mín:     {latency_metrics['min_ms']:.2f} ms")
    print(f"Latencia máx:     {latency_metrics['max_ms']:.2f} ms")
    print(f"Percentil 95:     {latency_metrics['p95_ms']:.2f} ms")
    print(f"Percentil 99:     {latency_metrics['p99_ms']:.2f} ms")
    print(f"Throughput (FPS): {latency_metrics['fps']:.2f}")
    print(f"{'='*50}")

    with open(RESULTS_DIR / "latency_metrics.json", 'w') as f:
        json.dump(latency_metrics, f, indent=4)

    # Visualizar métricas (se pasa el objeto de validación para la matriz real)
    plot_metrics(metrics, latency_metrics, val_results)

    return metrics, latency_metrics


def compute_confusion_matrix(model, images_dir, labels_dir, class_names, *,
                             conf=0.25, iou=0.45, device="cpu"):
    """
    Calcula una matriz de confusión para detección de objetos.

    Semántica (K = número de clases; el índice K-1 representa el fondo):
        cm[real][predicho]
        - cm[i][j] con i, j < K-1: instancia real de la clase i detectada como j.
        - cm[i][K-1]: instancia real de la clase i que no fue detectada (falso negativo).
        - cm[K-1][j]: detección de la clase j sin objeto real asociado (falso positivo).

    El emparejamiento es voraz y guiado por la confianza: cada predicción se
    asocia con la instancia real no emparejada de mayor intersección sobre unión
    (IoU) que supere el umbral. De este modo la suma de cada fila coincide con el
    número de instancias reales de esa clase, y la matriz es directamente
    interpretable.
    """
    K = len(class_names) + 1
    cm = np.zeros((K, K), dtype=int)

    def leer_gt(label_path, w, h):
        cajas = []
        if not label_path.exists():
            return cajas
        for linea in label_path.read_text().splitlines():
            if not linea.strip():
                continue
            clase, xc, yc, nw, nh = linea.split()[:5]
            xc, yc, nw, nh = float(xc), float(yc), float(nw), float(nh)
            cajas.append((
                int(clase),
                [(xc - nw / 2) * w, (yc - nh / 2) * h,
                 (xc + nw / 2) * w, (yc + nh / 2) * h],
            ))
        return cajas

    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    for img_path in sorted(images_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt = leer_gt(labels_dir / f"{img_path.stem}.txt", w, h)
        pred = model.predict(img, conf=conf, iou=iou, device=device, verbose=False)[0]

        detecciones = sorted(
            ((int(b.cls[0]), float(b.conf[0]),
              [float(v) for v in b.xyxy[0].cpu().numpy()]) for b in pred.boxes),
            key=lambda d: -d[1],
        )

        emparejado_gt = [False] * len(gt)

        for clase_pred, _conf, caja_pred in detecciones:
            mejor, mejor_iou = -1, iou
            for idx, (_clase_gt, caja_gt) in enumerate(gt):
                if emparejado_gt[idx]:
                    continue
                valor = _iou(caja_gt, caja_pred)
                if valor >= mejor_iou:
                    mejor, mejor_iou = idx, valor
            if mejor >= 0:
                emparejado_gt[mejor] = True
                cm[gt[mejor][0]][clase_pred] += 1
            else:
                cm[K - 1][clase_pred] += 1

        for idx, (clase_gt, _caja_gt) in enumerate(gt):
            if not emparejado_gt[idx]:
                cm[clase_gt][K - 1] += 1

    return cm


def plot_metrics(metrics, latency_metrics, val_results=None):
    """
    Genera gráficas de métricas de desempeño y latencia.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Métricas de Desempeño - Inspección Visual con YOLOv8", 
                 fontsize=14, fontweight='bold')

    # 1. Barras de métricas principales
    ax1 = axes[0, 0]
    main_metrics = ['mAP50', 'mAP50_95', 'precision', 'recall', 'mean_f1']
    values = [metrics[m] for m in main_metrics]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax1.bar(main_metrics, values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Valor", fontweight='bold')
    ax1.set_title("Métricas Principales", fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Distribución de latencias (mediciones reales, no simuladas)
    ax2 = axes[0, 1]
    latencies_data = latency_metrics.get("muestras_ms", [])
    if latencies_data:
        ax2.hist(latencies_data, bins=min(30, max(5, len(latencies_data) // 4)),
                 color='#3498db', edgecolor='black', alpha=0.7)
    ax2.axvline(latency_metrics['mean_ms'], color='red', linestyle='--', 
                linewidth=2, label=f"Media: {latency_metrics['mean_ms']:.2f} ms")
    ax2.axvline(latency_metrics['p95_ms'], color='orange', linestyle='--', 
                linewidth=2, label=f"P95: {latency_metrics['p95_ms']:.2f} ms")
    ax2.set_xlabel("Latencia (ms)", fontweight='bold')
    ax2.set_ylabel("Frecuencia", fontweight='bold')
    ax2.set_title("Distribución de Latencia de Inferencia", fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Matriz de confusión real producida por la validación de YOLO
    ax3 = axes[1, 0]
    if val_results is not None and getattr(val_results, "confusion_matrix", None) is not None:
        cm = np.asarray(val_results.confusion_matrix.matrix)
        nombres = [val_results.names[i] for i in sorted(val_results.names)] + ["fondo"]
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=ax3,
                    xticklabels=nombres, yticklabels=nombres)
        ax3.set_xlabel("Predicho", fontweight='bold')
        ax3.set_ylabel("Real", fontweight='bold')
        ax3.set_title("Matriz de Confusión (conjunto de prueba)", fontweight='bold')
    else:
        ax3.text(0.5, 0.5, "Matriz no disponible", ha='center', va='center')
        ax3.set_title("Matriz de Confusión", fontweight='bold')
        ax3.axis('off')

    # 4. Métricas de latencia
    ax4 = axes[1, 1]
    latency_vals = [latency_metrics['min_ms'], latency_metrics['mean_ms'], 
                    latency_metrics['median_ms'], latency_metrics['p95_ms'], 
                    latency_metrics['max_ms']]
    latency_labels = ['Mín', 'Media', 'Mediana', 'P95', 'Máx']
    colors_lat = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    bars2 = ax4.bar(latency_labels, latency_vals, color=colors_lat, 
                    edgecolor='black', linewidth=1.2)
    ax4.set_ylabel("Latencia (ms)", fontweight='bold')
    ax4.set_title("Resumen de Latencias", fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, latency_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metrics_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Dashboard de métricas guardado en: {RESULTS_DIR / 'metrics_dashboard.png'}")


# ==============================================================================
# FASE 5: INSPECCIÓN EN TIEMPO REAL
# ==============================================================================

def inspect_image(image_path, model_path, save_result=True, model=None):
    """
    Inspecciona una imagen y determina si la pieza es APROBADA o RECHAZADA.

    Args:
        image_path: ruta de la imagen a inspeccionar.
        model_path: ruta del modelo (se usa solo si no se proporciona ``model``).
        save_result: si es True, guarda la imagen anotada en results/inspections/.
        model: instancia de ``YOLO`` ya cargada. Permite reutilizar el modelo en
            inspecciones sucesivas; recargarlo en cada imagen domina el tiempo de
            proceso y falsea las métricas de latencia del lote.

    Returns:
        dict: Resultado de la inspección con decisión, confianza y métricas.
    """
    if model is None:
        model = YOLO(model_path)

    # Cargar imagen
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    h, w = img.shape[:2]

    # Inferencia y decisión: la lógica vive en `src/inspection.py` y es la misma
    # que emplea la aplicación web, de modo que ambas no pueden divergir.
    resultado = inspeccionar(model, img)

    decision = resultado.decision
    confianza = resultado.confianza
    margen = resultado.margen_aprobacion
    inference_time = resultado.latencia_ms
    decision_color = (0, 255, 0) if resultado.aprobada else (0, 0, 255)

    defects_found = [
        {"tipo": d.tipo, "confianza": d.confianza, "bbox": d.bbox}
        for d in resultado.defectos
    ]

    # Dibujar resultados en la imagen
    result_img = img.copy()

    # Dibujar bounding boxes
    for defect in defects_found:
        x1, y1, x2, y2 = defect["bbox"]
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 0, 255), 2)
        label = f"{defect['tipo']}: {defect['confianza']:.2f}"
        cv2.putText(result_img, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Panel de decisión
    panel_h = 80
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:, :] = decision_color

    text = f"DECISIÓN: {decision} | Confianza: {confianza:.2%} | Latencia: {inference_time:.1f}ms"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (panel_h + text_size[1]) // 2
    cv2.putText(panel, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, (255, 255, 255), 2)

    # Combinar
    final_img = np.vstack([result_img, panel])

    # Guardar
    if save_result:
        INSPECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = INSPECTIONS_DIR / f"inspection_{timestamp}.png"
        cv2.imwrite(str(save_path), final_img)

    inspection_result = {
        "imagen": str(image_path),
        "decision": decision,
        "confianza": confianza,
        "margen_aprobacion": margen,
        "latencia_ms": inference_time,
        "defectos_detectados": len(defects_found),
        "detalles_defectos": defects_found,
        "timestamp": datetime.now().isoformat(),
    }

    return inspection_result, final_img


def batch_inspection(test_dir, model_path, max_images=20):
    """
    Realiza inspección en lote sobre un directorio de imágenes.

    La selección de imágenes es estratificada por clase para que el lote
    reproduzca la composición del conjunto de prueba; de lo contrario, el orden
    alfabético de los archivos puede concentrar todos los defectos al principio
    y sesgar por completo las tasas de aprobación.
    """
    print("\n" + "="*70)
    print("FASE 5: INSPECCIÓN EN LOTE")
    print("="*70)

    test_path = Path(test_dir)
    label_path = test_path.parent.parent / "labels" / "test"

    todas = sorted(list(test_path.glob("*.png")) + list(test_path.glob("*.jpg")))

    # Agrupar por la clase anotada; las imágenes sin anotación son 'good'.
    por_clase = {}
    for img_path in todas:
        label = label_path / f"{img_path.stem}.txt"
        contenido = label.read_text().strip() if label.exists() else ""
        clase = contenido.split()[0] if contenido else "0"
        por_clase.setdefault(clase, []).append(img_path)

    # Muestreo proporcional al tamaño de cada clase, mediante reparto por
    # restos mayores para que el lote alcance exactamente el tamaño solicitado.
    seleccion = []
    if len(todas) > max_images:
        rng = random.Random(CONFIG["seed"])
        for rutas in por_clase.values():
            rng.shuffle(rutas)

        cuotas = {}
        restos = []
        asignado = 0
        for clase, rutas in sorted(por_clase.items()):
            exacto = len(rutas) / len(todas) * max_images
            cuotas[clase] = min(len(rutas), int(exacto))
            asignado += cuotas[clase]
            restos.append((exacto - int(exacto), clase))

        for _resto, clase in sorted(restos, key=lambda x: -x[0]):
            if asignado >= max_images:
                break
            if cuotas[clase] < len(por_clase[clase]):
                cuotas[clase] += 1
                asignado += 1

        for clase, rutas in sorted(por_clase.items()):
            seleccion.extend(rutas[:cuotas[clase]])

        # Relleno de seguridad si alguna clase no tenía suficientes ejemplares.
        if len(seleccion) < max_images:
            elegidos = set(seleccion)
            for ruta in todas:
                if len(seleccion) >= max_images:
                    break
                if ruta not in elegidos:
                    seleccion.append(ruta)

        seleccion = sorted(seleccion)[:max_images]
    else:
        seleccion = todas[:max_images]

    image_files = seleccion

    print(f"[INFO] Inspeccionando {len(image_files)} imágenes...")

    # El modelo se carga una sola vez: recargarlo por imagen dominaría el tiempo
    # de proceso y haría que las latencias del lote no reflejaran la inferencia.
    model = YOLO(model_path)

    results = []
    for img_path in image_files:
        result, _ = inspect_image(img_path, model_path, save_result=True, model=model)
        results.append(result)
        detalle_conf = (f"conf: {result['confianza']:.2%}" if result["decision"] == "RECHAZADA"
                        else f"margen: {result['margen_aprobacion']:.2%}")
        print(f"  {[img_path.name][0]}: {result['decision']:>9s} "
              f"({detalle_conf}, lat: {result['latencia_ms']:.1f}ms, "
              f"defectos: {result['defectos_detectados']})")

    # Resumen
    approved = sum(1 for r in results if r["decision"] == "APROBADA")
    rejected = len(results) - approved
    avg_latency = np.mean([r["latencia_ms"] for r in results])

    # Desglose por tipo de pieza. La clase real se obtiene del archivo de
    # etiquetas: la conversión a formato YOLO renombra las imágenes con índices
    # correlativos, de modo que el nombre del archivo no identifica la clase.
    classes_inv = {0: "good", 1: "broken_large", 2: "broken_small", 3: "contamination"}
    por_tipo = {}
    for r in results:
        ruta = Path(r["imagen"])
        label = label_path / f"{ruta.stem}.txt"
        contenido = label.read_text().strip() if label.exists() else ""
        clase = classes_inv.get(int(contenido.split()[0]) if contenido else 0, "desconocida")
        datos = por_tipo.setdefault(clase, {"total": 0, "aprobadas": 0, "rechazadas": 0})
        datos["total"] += 1
        datos["aprobadas" if r["decision"] == "APROBADA" else "rechazadas"] += 1

    print(f"\n{'='*50}")
    print("RESUMEN DE INSPECCIÓN EN LOTE")
    print(f"{'='*50}")
    print(f"Total analizadas: {len(results)}")
    print(f"Aprobadas:        {approved} ({approved/len(results)*100:.1f}%)")
    print(f"Rechazadas:       {rejected} ({rejected/len(results)*100:.1f}%)")
    print(f"Latencia media:   {avg_latency:.2f} ms")
    print(f"Throughput:       {1000/avg_latency:.1f} FPS")
    print(f"{'='*50}")

    # Guardar reporte
    report = {
        "total_inspeccionadas": len(results),
        "aprobadas": approved,
        "rechazadas": rejected,
        "latencia_media_ms": float(avg_latency),
        "throughput_fps": float(1000/avg_latency),
        "por_tipo": por_tipo,
        "detalles": results
    }

    with open(RESULTS_DIR / "batch_inspection_report.json", 'w') as f:
        json.dump(report, f, indent=4, default=str)

    print(f"[INFO] Reporte guardado en: {RESULTS_DIR / 'batch_inspection_report.json'}")

    return report


# ==============================================================================
# FASE 6: EXPORTACIÓN Y DESPLIEGUE
# ==============================================================================

def export_model(model_path):
    """
    Exporta el modelo a formatos de despliegue (ONNX, TorchScript, OpenVINO).
    """
    print("\n" + "="*70)
    print("FASE 6: EXPORTACIÓN DEL MODELO")
    print("="*70)

    model = YOLO(model_path)

    export_formats = ["onnx", "torchscript"]
    export_dir = EXPORTS_DIR
    export_dir.mkdir(exist_ok=True)

    for fmt in export_formats:
        try:
            print(f"[INFO] Exportando a formato: {fmt.upper()}")
            model.export(format=fmt, imgsz=CONFIG["img_size"], 
                        project=str(export_dir), name=fmt)
            print(f"[INFO] Exportación {fmt.upper()} exitosa")
        except Exception as e:
            print(f"[WARN] Error exportando {fmt}: {e}")

    print(f"[INFO] Modelos exportados en: {export_dir}")


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def evaluar_y_exportar():
    """
    Ejecuta únicamente las fases de evaluación, inspección por lotes y
    exportación, reutilizando el modelo ya entrenado.

    Resulta útil cuando se necesita recalcular métricas o artefactos sin repetir
    el entrenamiento completo.
    """
    data_yaml = str(DATA_YAML)
    model_path = trained_model()

    if not model_path.exists():
        print(f"[ERROR] No se encontró el modelo entrenado en: {model_path}")
        print("[INFO] Ejecute primero el pipeline completo.")
        return

    print("="*70)
    print("EVALUACIÓN Y EXPORTACIÓN (REUTILIZANDO MODELO ENTRENADO)")
    print("="*70)

    evaluate_model(str(model_path), data_yaml)
    batch_inspection(YOLO_DATASET_DIR / "images" / "test", str(model_path), max_images=30)
    export_model(str(model_path))

    print("\n" + "="*70)
    print("EVALUACIÓN Y EXPORTACIÓN COMPLETADAS")
    print("="*70)


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    """
    Ejecuta el pipeline completo del proyecto.
    """
    print("="*70)
    print("PROYECTO: VISIÓN ARTIFICIAL PARA CONTROL DE CALIDAD")
    print("Aplicaciones Prácticas de IA - YOLOv8 + MVTec AD")
    print("="*70)

    # Fase 1: Preparación
    if not prepare_mvtec_dataset():
        print("[ERROR] No se pudo preparar el dataset. Abortando.")
        return

    # Fase 2: Exploración
    explore_dataset()

    # Fase 3: Conversión y Entrenamiento
    data_yaml = convert_to_yolo_format()
    model_path = train_model(data_yaml)

    # Fase 4: Evaluación
    metrics, latency = evaluate_model(model_path, data_yaml)

    # Fase 5: Inspección en lote
    test_dir = Path(str(YOLO_DATASET_DIR / "images/test"))
    batch_inspection(test_dir, model_path, max_images=30)

    # Fase 6: Exportación
    export_model(model_path)

    print("\n" + "="*70)
    print("PROYECTO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"[INFO] Resultados guardados en: {RESULTS_DIR}")
    print(f"[INFO] Modelo entrenado en: {TRAINED_DIR}")
    print("\nPara ejecutar la aplicación web, use: streamlit run scripts/app.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline de inspección visual con YOLOv8 y MVTec AD."
    )
    parser.add_argument(
        "fase",
        nargs="?",
        default="completo",
        choices=["completo", "evaluar"],
        help=("'completo' ejecuta todas las fases; 'evaluar' recalcula métricas, "
              "inspección por lotes y exportación reutilizando el modelo entrenado."),
    )
    args = parser.parse_args()

    if args.fase == "evaluar":
        evaluar_y_exportar()
    else:
        main()
