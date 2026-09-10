"""
================================================================================
LÓGICA DE INSPECCIÓN COMPARTIDA
================================================================================

Este módulo concentra la decisión de aprobación o rechazo de una pieza.

`inspeccion_visual_yolov8.py` (pipeline) y `app.py` (interfaz web) mantenían dos
implementaciones paralelas de la misma regla, con umbrales y etiquetas
duplicados. Cualquier cambio en los criterios de decisión debía replicarse en
ambos lugares, con el riesgo de que divergieran silenciosamente.

La inferencia se ejecuta con un umbral de confianza muy bajo y la decisión se
aplica después, en el post-procesado. Ese diseño, con una sola pasada de
inferencia, permite informar además la holgura real de una aprobación: la
distancia entre la mejor detección de defecto y el umbral configurado.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import CLASS_LABELS, CLASS_NAMES, CONFIG

# Umbral mínimo con el que se ejecuta la inferencia. La decisión se toma después
# aplicando CONFIG["confidence_threshold"], de modo que una pieza aprobada
# conserve información sobre el margen que la separa del rechazo.
INFERENCE_FLOOR = 0.01


@dataclass
class Defecto:
    """Un defecto detectado por encima del umbral de decisión."""

    tipo: str
    etiqueta: str
    confianza: float
    bbox: list[float]


@dataclass
class ResultadoInspeccion:
    """Resultado de inspeccionar una pieza."""

    decision: str                       # "APROBADA" | "RECHAZADA"
    aprobada: bool
    confianza: float                    # confianza del defecto que motiva el rechazo, o 0
    margen_aprobacion: float            # holgura de la aprobación respecto del umbral
    latencia_ms: float
    defectos: list[Defecto] = field(default_factory=list)

    @property
    def num_defectos(self) -> int:
        return len(self.defectos)


def analizar_predicciones(boxes, umbral: float) -> tuple[list[Defecto], float]:
    """
    Separa las detecciones en defectos confirmados y descartados por umbral.

    Args:
        boxes: colección de cajas de Ultralytics (`results.boxes`).
        umbral: umbral de confianza para aceptar un defecto.

    Returns:
        (defectos confirmados, confianza máxima descartada por quedar bajo el umbral)
    """
    defectos: list[Defecto] = []
    max_bajo_umbral = 0.0

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # La clase 0 es 'good': su presencia no constituye un defecto.
        if cls_id == 0:
            continue

        tipo = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "desconocido"

        if conf < umbral:
            max_bajo_umbral = max(max_bajo_umbral, conf)
            continue

        defectos.append(Defecto(
            tipo=tipo,
            etiqueta=CLASS_LABELS.get(tipo, tipo),
            confianza=conf,
            bbox=[float(v) for v in box.xyxy[0].cpu().numpy()],
        ))

    return defectos, max_bajo_umbral


def decidir(defectos: list[Defecto], max_bajo_umbral: float, umbral: float) -> tuple[str, bool, float, float]:
    """
    Aplica la regla de decisión del sistema.

    Una pieza se rechaza si se confirmó al menos un defecto. En caso contrario se
    aprueba y se informa el margen respecto del umbral.

    Returns:
        (decisión, aprobada, confianza, margen_aprobacion)
    """
    if defectos:
        principal = max(defectos, key=lambda d: d.confianza)
        return "RECHAZADA", False, principal.confianza, 0.0

    # Margen = cuánto le faltó a la mejor detección para superar el umbral.
    # Una holgura amplia indica una aprobación más segura.
    return "APROBADA", True, 0.0, umbral - max_bajo_umbral


def inspeccionar(model, imagen, *, conf: float | None = None, iou: float | None = None,
                 device: str | None = None) -> ResultadoInspeccion:
    """
    Ejecuta la inspección completa de una imagen.

    Args:
        model: instancia de ``YOLO`` ya cargada. Se recibe desde fuera para poder
            reutilizarla entre inspecciones: recargarla en cada imagen domina el
            tiempo de proceso e invalida las métricas de latencia por lote.
        imagen: imagen en formato BGR (``numpy.ndarray``).
        conf: umbral de confianza de decisión. Por defecto, el de ``CONFIG``.
        iou: umbral de IoU para la supresión de no máximos.
        device: dispositivo de inferencia.

    Returns:
        ResultadoInspeccion con la decisión y sus métricas asociadas.
    """
    import time

    umbral = CONFIG["confidence_threshold"] if conf is None else conf
    iou = CONFIG["iou_threshold"] if iou is None else iou
    device = CONFIG["device"] if device is None else device

    inicio = time.perf_counter()
    prediccion = model.predict(
        imagen,
        conf=INFERENCE_FLOOR,
        iou=iou,
        device=device,
        verbose=False,
    )[0]
    latencia_ms = (time.perf_counter() - inicio) * 1000

    defectos, max_bajo_umbral = analizar_predicciones(prediccion.boxes, umbral)
    decision, aprobada, confianza, margen = decidir(defectos, max_bajo_umbral, umbral)

    return ResultadoInspeccion(
        decision=decision,
        aprobada=aprobada,
        confianza=confianza,
        margen_aprobacion=margen,
        latencia_ms=latencia_ms,
        defectos=defectos,
    )
