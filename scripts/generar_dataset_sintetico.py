"""
================================================================================
GENERADOR DE DATASET SINTÉTICO TIPO MVTec AD — CATEGORÍA "bottle"
================================================================================

Motivación metodológica
-----------------------
El conjunto MVTec AD está diseñado para detección de anomalías NO supervisada:
su partición de entrenamiento contiene únicamente imágenes normales (good). Un
detector supervisado como YOLOv8, en cambio, requiere ejemplos positivos de cada
clase de defecto para aprender a localizarlos. Por esa razón se construye una
réplica sintética de la categoría bottle con defectos etiquetados, respetando la
estructura de directorios del MVTec AD original.

Estructura generada
-------------------
    data/mvtec_ad/bottle/
        train/good/                       imágenes normales (fondo, sin cajas)
        train/broken_large/               grietas grandes
        train/broken_small/               microfracturas
        train/contamination/              contaminación superficial
        test/good/
        test/broken_large/
        test/broken_small/
        test/contamination/
        ground_truth/<defecto>/<img>_mask.png

Uso
---
    python scripts/generar_dataset_sintetico.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Permite importar el paquete `src` al ejecutar el script desde cualquier lugar.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG, MVTEC_DIR  # noqa: E402

# ------------------------------------------------------------------------------
# PARÁMETROS
# ------------------------------------------------------------------------------
ROOT = MVTEC_DIR / CONFIG["category"]
IMGSZ = 512
SEED = 2026

TRAIN_GOOD = 110
TRAIN_DEFECT = 32      # por clase de defecto
TEST_GOOD = 30
TEST_DEFECT = 16       # por clase de defecto

DEFECTOS = ("broken_large", "broken_small", "contamination")


# ------------------------------------------------------------------------------
# FONDO Y BOTELLA
# ------------------------------------------------------------------------------
def _fondo(rng: np.random.Generator) -> np.ndarray:
    """Banda transportadora industrial con textura e iluminación variable."""
    tono = int(rng.integers(58, 92))
    img = np.full((IMGSZ, IMGSZ, 3), tono, np.float32)
    img[..., 1] += float(rng.uniform(2.0, 9.0))
    img[..., 2] += float(rng.uniform(4.0, 12.0))

    ruido = rng.normal(0.0, 6.0, (IMGSZ, IMGSZ, 1)).astype(np.float32)
    img = img + ruido

    grad = np.linspace(-12.0, 12.0, IMGSZ, dtype=np.float32)[:, None, None]
    return np.clip(img + grad, 0, 255).astype(np.uint8)


def _botella(rng: np.random.Generator):
    """Dibuja una botella cilíndrica con sombreado realista."""
    mask = np.zeros((IMGSZ, IMGSZ), np.uint8)

    cx = IMGSZ // 2 + int(rng.integers(-14, 15))
    ancho = int(rng.integers(132, 172))
    y_top = int(rng.integers(150, 178))
    y_bot = int(rng.integers(452, 486))
    cuello_ancho = int(rng.integers(52, 72))
    y_cuello = y_top - int(rng.integers(62, 88))

    x0, x1 = cx - ancho // 2, cx + ancho // 2

    # Cuello
    cv2.rectangle(
        mask,
        (cx - cuello_ancho // 2, y_cuello),
        (cx + cuello_ancho // 2, y_top + 6),
        255,
        -1,
    )
    # Cuerpo + tapas curvas
    cv2.rectangle(mask, (x0, y_top), (x1, y_bot), 255, -1)
    cv2.ellipse(mask, (cx, y_bot), (ancho // 2, 16), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (cx, y_top), (ancho // 2, 16), 0, 0, 360, 255, -1)

    base = np.array([214.0, 218.0, 224.0], np.float32)
    base *= float(rng.uniform(0.90, 1.02))

    # Sombreado cilíndrico: perfil parabólico + banda especular
    X = np.arange(IMGSZ, dtype=np.float32)[None, :].repeat(IMGSZ, 0)
    centro_luz = x0 + (x1 - x0) * float(rng.uniform(0.30, 0.52))
    semiancho = max(1.0, (x1 - x0) / 2.0)

    t = (X - centro_luz) / semiancho
    perfil = np.clip(1.0 - 0.40 * t**2, 0.42, 1.0)
    brillo = np.exp(-((X - centro_luz) / max(1.0, (x1 - x0) * 0.11)) ** 2) * 0.30
    factor = perfil + brillo

    cuerpo = base[None, None, :] * factor[..., None]

    Y = np.arange(IMGSZ, dtype=np.float32)[:, None].repeat(IMGSZ, 1)
    atenuacion = 1.0 - 0.10 * (Y - y_top) / max(1.0, y_bot - y_top)
    cuerpo = cuerpo * atenuacion[..., None]

    img = _fondo(rng).astype(np.float32)
    dentro = mask[:, :, None] > 0
    img = np.where(dentro, np.clip(cuerpo, 0, 255), img)

    return img.astype(np.uint8), mask, (cx, y_top, y_bot)


# ------------------------------------------------------------------------------
# DEFECTOS
# ------------------------------------------------------------------------------
def _trazo_grieta(rng: np.random.Generator, cx: float, cy: float, longitud: float, n: int):
    """Genera una polilínea irregular que simula una fractura."""
    pts = [(float(cx), float(cy))]
    ang = float(rng.uniform(-np.pi, np.pi))
    for _ in range(n):
        ang += float(rng.uniform(-0.55, 0.55))
        paso = longitud / n
        cx += np.cos(ang) * paso
        cy += np.sin(ang) * paso
        pts.append((float(cx), float(cy)))
    return np.array(pts, np.int32)


def _capa_mancha(rng: np.random.Generator, cx: int, cy: int, radio: int) -> np.ndarray:
    """Genera una mancha irregular compuesta por múltiples blobs."""
    capa = np.zeros((IMGSZ, IMGSZ), np.uint8)
    for _ in range(int(rng.integers(14, 34))):
        dx = int(rng.normal(0.0, radio * 0.55))
        dy = int(rng.normal(0.0, radio * 0.55))
        r = int(rng.integers(max(2, radio // 8), max(4, radio // 3)))
        cv2.circle(capa, (cx + dx, cy + dy), r, 255, -1)
    return capa


def _aplicar_defecto(
    img: np.ndarray,
    rng: np.random.Generator,
    tipo: str,
    cx: int,
    y_top: int,
    y_bot: int,
) -> np.ndarray:
    """Dibuja el defecto sobre la imagen y devuelve su máscara binaria."""
    mask = np.zeros((IMGSZ, IMGSZ), np.uint8)
    cy = int(rng.integers(y_top + 40, y_bot - 40))

    if tipo == "broken_large":
        largo = float(rng.integers(120, 230))
        pts = _trazo_grieta(rng, cx + int(rng.integers(-40, 41)), cy, largo, 7)
        grosor = int(rng.integers(9, 16))
        cv2.polylines(mask, [pts], False, 255, grosor)
        cv2.polylines(img, [pts], False, (152, 154, 160), grosor + 8)  # halo
        cv2.polylines(img, [pts], False, (70, 72, 78), grosor)         # núcleo

    elif tipo == "broken_small":
        largo = float(rng.integers(30, 68))
        pts = _trazo_grieta(rng, cx + int(rng.integers(-55, 56)), cy, largo, 5)
        grosor = int(rng.integers(3, 6))
        cv2.polylines(mask, [pts], False, 255, grosor)
        cv2.polylines(img, [pts], False, (88, 90, 96), grosor)

    else:  # contamination
        radio = int(rng.integers(18, 42))
        capa = _capa_mancha(rng, cx + int(rng.integers(-45, 46)), cy, radio)
        mask = cv2.bitwise_or(mask, capa)
        img[capa > 0] = (58, 74, 112)

        extra = _capa_mancha(
            rng,
            cx + int(rng.integers(-60, 61)),
            cy + int(rng.integers(-40, 41)),
            max(6, radio // 2),
        )
        mask = cv2.bitwise_or(mask, extra)
        img[extra > 0] = (72, 88, 128)

    return mask


# ------------------------------------------------------------------------------
# GENERACIÓN
# ------------------------------------------------------------------------------
def _guardar(img, mask, dir_img: Path, dir_gt: Path, clase: str, nombre: str) -> None:
    dir_img.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dir_img / f"{nombre}.png"), img)
    if mask is not None and mask.any():
        destino = dir_gt / clase
        destino.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destino / f"{nombre}_mask.png"), mask)


def generar() -> dict:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    rng = np.random.default_rng(SEED)
    resumen = {}

    def normales(n: int, split: str) -> int:
        for i in range(n):
            img, _, _ = _botella(rng)
            _guardar(img, None, ROOT / split / "good", ROOT / "ground_truth", "good", f"good_{split}_{i:04d}")
        return n

    def defectuosas(n: int, split: str, tipo: str) -> int:
        for i in range(n):
            img, _, (cx, y_top, y_bot) = _botella(rng)
            mask = _aplicar_defecto(img, rng, tipo, cx, y_top, y_bot)
            _guardar(img, mask, ROOT / split / tipo, ROOT / "ground_truth", tipo, f"{tipo}_{split}_{i:04d}")
        return n

    resumen["train_good"] = normales(TRAIN_GOOD, "train")
    resumen["test_good"] = normales(TEST_GOOD, "test")

    for tipo in DEFECTOS:
        resumen[f"train_{tipo}"] = defectuosas(TRAIN_DEFECT, "train", tipo)
        resumen[f"test_{tipo}"] = defectuosas(TEST_DEFECT, "test", tipo)

    return resumen


def main() -> None:
    resumen = generar()
    print("=" * 70)
    print("DATASET SINTÉTICO GENERADO")
    print("=" * 70)
    print(f"Raíz: {ROOT.resolve()}")
    for clave, valor in resumen.items():
        print(f"  {clave:<24} {valor}")
    print(f"  {'TOTAL':<24} {sum(resumen.values())}")
    print("\nImágenes de entrenamiento con defecto:", sum(v for k, v in resumen.items() if k.startswith("train_") and v and "good" not in k))
    print("Máscaras en:", (ROOT / "ground_truth").resolve())


if __name__ == "__main__":
    main()
