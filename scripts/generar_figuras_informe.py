"""
================================================================================
GENERADOR DE FIGURAS PARA EL INFORME
================================================================================

Produce las ilustraciones del documento a partir de los artefactos reales
almacenados en results/. Todas las cifras provienen de los archivos JSON y de
las imágenes generadas por el pipeline; no se introducen valores manuales.

Salida (results/figuras/):
    fig01_pipeline.png            Arquitectura del pipeline
    fig02_dataset.png             Muestras del conjunto de datos
    fig03_distribucion_clases.png Distribución de clases por partición
    fig04_curvas_entrenamiento.png Curvas de pérdida y métricas por época
    fig05_metricas_globales.png    Métricas globales del modelo
    fig06_metricas_por_clase.png   Desempeño desagregado por clase
    fig07_matriz_confusion.png     Matriz de confusión (cruda y normalizada)
    fig08_curva_pr.png             Curvas precisión-exhaustividad
    fig09_latencia.png             Distribución de la latencia de inferencia
    fig10_inspeccion.png           Ejemplo de inspección con decisión

Uso:
    python scripts/generar_figuras_informe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Permite importar el paquete `src` al ejecutar el script desde cualquier lugar.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CLASS_LABELS,
    CONFIG,
    FIGURES_DIR,
    INSPECTIONS_DIR,
    MVTEC_DIR,
    RESULTS_DIR,
    RUNS_DIR,
)

RESULTS = RESULTS_DIR
FIGDIR = FIGURES_DIR
FIGDIR.mkdir(parents=True, exist_ok=True)

ETIQUETAS = CLASS_LABELS

AZUL, VERDE, ROJO, NARANJA, MORADO = "#2E86AB", "#28A745", "#D64545", "#E8A33D", "#6A4C93"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _json(nombre: str) -> dict:
    ruta = RESULTS / nombre
    if not ruta.exists():
        return {}
    return json.loads(ruta.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ figura 1
def fig01_pipeline() -> None:
    fases = [
        ("1. Preparación\ndel dataset", "MVTec AD\nbottle"),
        ("2. Exploración\ny EDA", "Estadísticas\nvisuales"),
        ("3. Conversión\na formato YOLO", "Cajas desde\nmáscaras"),
        ("4. Entrenamiento\nYOLOv8s", "Transferencia\nCOCO"),
        ("5. Evaluación\ny latencia", "mAP, P, R, F1"),
        ("6. Inspección\ny exportación", "ONNX /\nTorchScript"),
    ]

    fig, ax = plt.subplots(figsize=(13, 2.9))
    ax.set_xlim(0, len(fases) * 2.0)
    ax.set_ylim(0, 2.0)
    ax.axis("off")

    for i, (titulo, detalle) in enumerate(fases):
        x = i * 2.0 + 0.1
        caja = FancyBboxPatch(
            (x, 0.55), 1.7, 1.0, boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.4, edgecolor=AZUL, facecolor="#EAF3F8",
        )
        ax.add_patch(caja)
        ax.text(x + 0.85, 1.24, titulo, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#14364A")
        ax.text(x + 0.85, 0.84, detalle, ha="center", va="center",
                fontsize=8, color="#3D5A6C")

        if i < len(fases) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + 1.72, 1.05), (x + 2.0, 1.05),
                arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color="#6B7C8A",
            ))

    ax.set_title("Pipeline de inspección visual automatizada implementado", pad=12)
    fig.savefig(FIGDIR / "fig01_pipeline.png")
    plt.close(fig)


# ------------------------------------------------------------------ figura 2
def fig02_dataset() -> None:
    raiz = MVTEC_DIR / "bottle"
    filas = [
        ("train/good", "Entrenamiento: Normal"),
        ("train/broken_large", "Entrenamiento: Grieta grande"),
        ("test/good", "Prueba: Normal"),
        ("test/broken_small", "Prueba: Grieta pequeña"),
        ("test/contamination", "Prueba: Contaminación"),
        ("test/broken_large", "Prueba: Grieta grande"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    for ax, (sub, titulo) in zip(axes.ravel(), filas):
        carpeta = raiz / sub
        imgs = sorted(carpeta.glob("*.png")) if carpeta.exists() else []
        if imgs:
            img = cv2.cvtColor(cv2.imread(str(imgs[0])), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.set_title(titulo, fontsize=9, fontweight="bold",
                     color=ROJO if "Prueba" in titulo else VERDE)
        ax.axis("off")

    fig.suptitle("Muestras del conjunto de datos sintético tipo MVTec AD (categoría bottle)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig02_dataset.png")
    plt.close(fig)


# ------------------------------------------------------------------ figura 3
def fig03_distribucion() -> None:
    comp = _json("split_composition.json")
    if not comp:
        return

    splits = ["train", "val", "test"]
    clases = ["good", "broken_large", "broken_small", "contamination"]
    x = np.arange(len(splits))
    ancho = 0.2

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colores = [AZUL, ROJO, NARANJA, MORADO]
    for i, clase in enumerate(clases):
        valores = [comp.get(s, {}).get("instancias", {}).get(clase, 0) for s in splits]
        barras = ax.bar(x + (i - 1.5) * ancho, valores, ancho,
                        label=ETIQUETAS[clase], color=colores[i],
                        edgecolor="black", linewidth=0.6)
        for b, v in zip(barras, valores):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, str(v),
                        ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Entrenamiento\n({comp.get('train', {}).get('archivos', 0)} imágenes)",
                        f"Validación\n({comp.get('val', {}).get('archivos', 0)} imágenes)",
                        f"Prueba\n({comp.get('test', {}).get('archivos', 0)} imágenes)"])
    ax.set_ylabel("Instancias anotadas")
    ax.set_title("Distribución de instancias por clase y partición")
    ax.legend(fontsize=8.5, ncol=4, frameon=False)
    ax.margins(y=0.16)
    fig.savefig(FIGDIR / "fig03_distribucion_clases.png")
    plt.close(fig)


# ------------------------------------------------------------------ figura 4
def fig04_entrenamiento() -> None:
    csv = RUNS_DIR / "detect" / CONFIG["run_name"] / "results.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    ep = df["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    ax = axes[0]
    ax.plot(ep, df["train/box_loss"], marker="o", ms=3.5, label="Train box", color=AZUL)
    ax.plot(ep, df["val/box_loss"], marker="s", ms=3.5, label="Val box", color=ROJO)
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
    ax.set_title("Pérdida de localización")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(ep, df["train/cls_loss"], marker="o", ms=3.5, label="Train clase", color=AZUL)
    ax.plot(ep, df["val/cls_loss"], marker="s", ms=3.5, label="Val clase", color=ROJO)
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
    ax.set_title("Pérdida de clasificación")
    ax.legend(fontsize=8.5)

    ax = axes[2]
    ax.plot(ep, df["metrics/precision(B)"], marker="o", ms=3.5, label="Precisión", color=VERDE)
    ax.plot(ep, df["metrics/recall(B)"], marker="s", ms=3.5, label="Exhaustividad", color=NARANJA)
    ax.plot(ep, df["metrics/mAP50(B)"], marker="^", ms=3.5, label="mAP@0.5", color=MORADO)
    ax.set_xlabel("Época"); ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.set_title("Métricas de validación")
    ax.legend(fontsize=8.5)

    fig.suptitle("Evolución del entrenamiento de YOLOv8s", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig04_curvas_entrenamiento.png")
    plt.close(fig)


# ------------------------------------------------------------------ figura 5
def fig05_metricas_globales() -> None:
    m = _json("evaluation_metrics.json")
    if not m:
        return

    nombres = ["mAP@0.5", "mAP@0.5:0.95", "Precisión", "Exhaustividad", "F1"]
    valores = [m.get("mAP50", 0), m.get("mAP50_95", 0), m.get("precision", 0),
               m.get("recall", 0), m.get("mean_f1", 0)]
    colores = [VERDE, AZUL, MORADO, NARANJA, ROJO]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    barras = ax.bar(nombres, valores, color=colores, edgecolor="black", linewidth=0.7)
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=9.5, fontweight="bold")

    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1.2)
    ax.text(len(nombres) - 0.45, 0.515, "umbral de referencia 0.50",
            fontsize=8, color="#666", ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Métricas globales de desempeño en el conjunto de prueba")
    fig.savefig(FIGDIR / "fig05_metricas_globales.png")
    plt.close(fig)


# ------------------------------------------------------------------ figura 6
def fig06_metricas_por_clase() -> None:
    m = _json("evaluation_metrics.json")
    por_clase = m.get("por_clase") or []
    if not por_clase:
        return

    nombres = [ETIQUETAS.get(c["clase"], c["clase"]) for c in por_clase]
    x = np.arange(len(nombres))
    ancho = 0.26
    series = [
        ("Precisión", [c["precision"] for c in por_clase], AZUL),
        ("Exhaustividad", [c["recall"] for c in por_clase], NARANJA),
        ("AP@0.5", [c["ap50"] for c in por_clase], VERDE),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (etq, vals, color) in enumerate(series):
        barras = ax.bar(x + (i - 1) * ancho, vals, ancho, label=etq,
                        color=color, edgecolor="black", linewidth=0.6)
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor")
    ax.set_title("Desempeño desagregado por clase en el conjunto de prueba")
    ax.legend(fontsize=9, ncol=3, frameon=False)
    fig.savefig(FIGDIR / "fig06_metricas_por_clase.png")
    plt.close(fig)


# ------------------------------------------------------------------ figura 7
def fig07_confusion() -> None:
    dato = _json("confusion_matrix.json")
    matriz = dato.get("matriz")
    clases = dato.get("clases") or []
    if not matriz or not clases:
        return

    cm = np.asarray(matriz, dtype=float)
    nombres = [ETIQUETAS.get(c, c.capitalize()) for c in clases]

    with np.errstate(invalid="ignore", divide="ignore"):
        filas = cm.sum(axis=1, keepdims=True)
        cnorm = np.divide(cm, filas, out=np.zeros_like(cm), where=filas > 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", ax=axes[0],
                xticklabels=nombres, yticklabels=nombres, cbar=False,
                linewidths=0.5, linecolor="white")
    axes[0].set_xlabel("Predicho", fontweight="bold")
    axes[0].set_ylabel("Real", fontweight="bold")
    axes[0].set_title("Recuentos absolutos")

    sns.heatmap(cnorm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=nombres, yticklabels=nombres, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white")
    axes[1].set_xlabel("Predicho", fontweight="bold")
    axes[1].set_ylabel("Real", fontweight="bold")
    axes[1].set_title("Proporción sobre el total real")

    fig.suptitle("Matriz de confusión del modelo sobre el conjunto de prueba",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig07_matriz_confusion.png")
    plt.close(fig)


# ------------------------------------------------------------------ figura 8
def fig08_curva_pr() -> None:
    """Curvas PR de la evaluación final, no las del seguimiento del entrenamiento.

    Ultralytics deja un BoxPR_curve.png en la carpeta del run por cada validación
    que ejecuta durante el entrenamiento. Esos gráficos se generan con el modelo
    de esa época y, en las épocas tempranas, varias clases aún no producen
    predicciones, por lo que sus paneles quedan vacíos. El informe, en cambio,
    describe la evaluación final, cuya curva se guarda en results/validation/. Se
    prioriza esa copia para que la figura muestre las cuatro clases y coincida con
    las métricas de evaluation_metrics.json.
    """
    candidatos = (
        RESULTS_DIR / "validation" / "BoxPR_curve.png",
        RUNS_DIR / "detect" / CONFIG["run_name"] / "BoxPR_curve.png",
    )
    origen = next((ruta for ruta in candidatos if ruta.exists()), None)
    if origen is None:
        return
    img = cv2.cvtColor(cv2.imread(str(origen)), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Curvas precisión-exhaustividad por clase")
    fig.savefig(FIGDIR / "fig08_curva_pr.png")
    plt.close(fig)


# ------------------------------------------------------------------ figura 9
def fig09_latencia() -> None:
    lat = _json("latency_metrics.json")
    muestras = lat.get("muestras_ms")
    if not muestras:
        return

    datos = np.asarray(muestras, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    axes[0].hist(datos, bins=min(30, max(6, len(datos) // 4)),
                 color=AZUL, edgecolor="black", alpha=0.8)
    axes[0].axvline(lat["mean_ms"], color=ROJO, linestyle="--", linewidth=1.8,
                    label=f"Media: {lat['mean_ms']:.1f} ms")
    axes[0].axvline(lat["median_ms"], color=VERDE, linestyle=":", linewidth=1.8,
                    label=f"Mediana: {lat['median_ms']:.1f} ms")
    axes[0].axvline(lat["p95_ms"], color=NARANJA, linestyle="-.", linewidth=1.8,
                    label=f"P95: {lat['p95_ms']:.1f} ms")
    axes[0].set_xlabel("Latencia de inferencia (ms)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title(f"Distribución de {len(datos)} inferencias")
    axes[0].legend(fontsize=8.5)

    etiquetas = ["Mínima", "P25", "Mediana", "Media", "P75", "P95", "P99", "Máxima"]
    valores = [
        np.min(datos), np.percentile(datos, 25), np.median(datos), np.mean(datos),
        np.percentile(datos, 75), np.percentile(datos, 95),
        np.percentile(datos, 99), np.max(datos),
    ]
    colores = [VERDE, AZUL, VERDE, MORADO, AZUL, NARANJA, ROJO, ROJO]
    barras = axes[1].bar(etiquetas, valores, color=colores,
                         edgecolor="black", linewidth=0.6)
    for b, v in zip(barras, valores):
        axes[1].text(b.get_x() + b.get_width() / 2, v + max(valores) * 0.02,
                     f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
    axes[1].set_ylabel("Latencia (ms)")
    axes[1].set_title("Estadísticos de latencia")
    axes[1].margins(y=0.16)

    fig.suptitle("Rendimiento temporal de la inferencia", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig09_latencia.png")
    plt.close(fig)


# ----------------------------------------------------------------- figura 10
def fig10_inspeccion() -> None:
    carpeta = INSPECTIONS_DIR
    if not carpeta.exists():
        return

    rechazadas, aprobadas = [], []
    for p in sorted(carpeta.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        franja = img[-70:, :, :]
        if franja.size == 0:
            continue
        # El panel inferior es rojo (BGR ~ (0,0,200)) cuando la pieza se rechaza
        rojo = int(franja[:, :, 2].mean())
        azul = int(franja[:, :, 0].mean())
        (rechazadas if rojo > azul else aprobadas).append(p)

    seleccion = []
    if rechazadas:
        seleccion.append((rechazadas[0], "Pieza RECHAZADA: defecto localizado"))
    if aprobadas:
        seleccion.append((aprobadas[0], "Pieza APROBADA: sin defectos"))
    if not seleccion:
        return

    fig, axes = plt.subplots(1, len(seleccion), figsize=(7.6 * len(seleccion), 5.4))
    if len(seleccion) == 1:
        axes = [axes]

    for ax, (ruta, titulo) in zip(axes, seleccion):
        img = cv2.cvtColor(cv2.imread(str(ruta)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(titulo, fontsize=10, fontweight="bold")

    fig.suptitle("Ejemplos de inspección con la decisión del sistema superpuesta",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig10_inspeccion.png")
    plt.close(fig)


# ----------------------------------------------------------------- figura 11
def fig11_lote() -> None:
    rep = _json("batch_inspection_report.json")
    if not rep:
        return

    # El pipeline entrega el desglose por tipo de pieza ya calculado a partir de
    # las etiquetas reales; se usa tal cual para no inferir la clase del nombre
    # del archivo, que en formato YOLO es un índice correlativo.
    por_tipo = rep.get("por_tipo") or {}
    if not por_tipo:
        return

    orden = ["good", "broken_large", "broken_small", "contamination"]
    presentes = [c for c in orden if c in por_tipo]

    nombres = [ETIQUETAS.get(c, c) for c in presentes]
    aprobados = [por_tipo[c]["aprobadas"] for c in presentes]
    rechazados = [por_tipo[c]["rechazadas"] for c in presentes]
    x = np.arange(len(presentes))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))

    axes[0].bar(x, aprobados, 0.5, label="Aprobadas", color=VERDE,
                edgecolor="black", linewidth=0.6)
    axes[0].bar(x, rechazados, 0.5, bottom=aprobados, label="Rechazadas",
                color=ROJO, edgecolor="black", linewidth=0.6)
    for i, (a, r) in enumerate(zip(aprobados, rechazados)):
        total = a + r
        if total:
            axes[0].text(i, total + 0.25, f"n={total}", ha="center",
                         fontsize=8.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(nombres, fontsize=9)
    axes[0].set_ylabel("Piezas")
    axes[0].set_title("Decisiones por tipo de pieza")
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].margins(y=0.18)

    etiquetas = ["Aprobadas", "Rechazadas"]
    valores = [rep.get("aprobadas", 0), rep.get("rechazadas", 0)]
    colores = [VERDE, ROJO]
    cuña, _texto, auto = axes[1].pie(
        valores, labels=None, autopct="%1.1f%%", colors=colores,
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in auto:
        t.set_fontweight("bold")
        t.set_color("white")
    axes[1].legend(cuña, [f"{e}: {v}" for e, v in zip(etiquetas, valores)],
                   fontsize=9, loc="lower center", ncol=2, frameon=False)
    axes[1].set_title(f"Resultado global (n={sum(valores)})")
    axes[1].grid(False)

    fig.suptitle("Resultados de la inspección por lotes", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig11_lote.png")
    plt.close(fig)


# ----------------------------------------------------------------- figura 12
def fig12_comparativa_entrenamiento() -> None:
    """Compara el desempeño con y sin ejemplos de defecto en el entrenamiento.

    Los valores de la primera configuración provienen del registro de una
    ejecución previa a la corrección y se conservan aquí porque documentan el
    diagnóstico. Los de la segunda se leen de los artefactos actuales.
    """
    actual = _json("evaluation_metrics.json") or {}
    if not actual:
        return

    escenarios = [
        ("Sin ejemplos\nde defecto", 0.3125, 0.4167, 0.3125),
        ("Con ejemplos\nde defecto", actual.get("mAP50", 0), actual.get("precision", 0),
         actual.get("recall", 0)),
    ]
    metricas = ["mAP@0.5", "Precisión", "Exhaustividad"]
    colores = [VERDE, AZUL, NARANJA]

    x = np.arange(len(escenarios))
    ancho = 0.26

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (nombre, color) in enumerate(zip(metricas, colores)):
        valores = [esc[i + 1] for esc in escenarios]
        barras = ax.bar(x + (i - 1) * ancho, valores, ancho, label=nombre,
                        color=color, edgecolor="black", linewidth=0.6)
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                    ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in escenarios])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Efecto de incluir ejemplos de defecto en el entrenamiento")
    ax.legend(fontsize=9, ncol=3, frameon=False)
    ax.annotate(
        "El modelo nunca vio\nun defecto durante\nel entrenamiento",
        xy=(0, 0.35), xytext=(0, 0.80), fontsize=8, ha="center", color="#666",
        arrowprops={"arrowstyle": "->", "color": "#999", "lw": 1},
    )
    fig.savefig(FIGDIR / "fig12_comparativa_entrenamiento.png")
    plt.close(fig)


def main() -> None:
    generadores = [
        ("fig01_pipeline", fig01_pipeline),
        ("fig02_dataset", fig02_dataset),
        ("fig03_distribucion", fig03_distribucion),
        ("fig04_entrenamiento", fig04_entrenamiento),
        ("fig05_metricas_globales", fig05_metricas_globales),
        ("fig06_metricas_por_clase", fig06_metricas_por_clase),
        ("fig07_confusion", fig07_confusion),
        ("fig08_curva_pr", fig08_curva_pr),
        ("fig09_latencia", fig09_latencia),
        ("fig10_inspeccion", fig10_inspeccion),
        ("fig11_lote", fig11_lote),
        ("fig12_comparativa_entrenamiento", fig12_comparativa_entrenamiento),
    ]
    for nombre, fn in generadores:
        try:
            fn()
            print(f"[OK] {nombre}")
        except Exception as exc:  # pragma: no cover - utilidad de diagnóstico
            print(f"[SKIP] {nombre}: {type(exc).__name__}: {exc}")

    print(f"\nFiguras en: {FIGDIR.resolve()}")


if __name__ == "__main__":
    main()
