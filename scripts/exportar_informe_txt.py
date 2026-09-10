"""
================================================================================
EXPORTADOR DEL INFORME WORD A TEXTO PLANO
================================================================================

Genera 'proyecto_inspeccion_visual.txt' a partir de
'proyecto_inspeccion_visual.docx'. De este modo el documento Word y la versión
en texto plano comparten una única fuente de verdad y no pueden divergir.

Uso:
    python scripts/exportar_informe_txt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

# Permite importar el paquete `src` al ejecutar el script desde cualquier lugar.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORT_DOCX, REPORT_TXT  # noqa: E402

ENTRADA = REPORT_DOCX
SALIDA = REPORT_TXT

ANCHO = 80


def _bloques(doc: Document):
    """Recorre párrafos y tablas en el orden real del documento."""
    cuerpo = doc.element.body
    for hijo in cuerpo.iterchildren():
        if hijo.tag.endswith("}p"):
            yield Paragraph(hijo, doc)
        elif hijo.tag.endswith("}tbl"):
            yield Table(hijo, doc)


def _regla(caracter: str = "=") -> str:
    return caracter * ANCHO


def _centrar(texto: str) -> str:
    return texto.center(ANCHO)


def exportar() -> None:
    doc = Document(ENTRADA)
    lineas: list[str] = []

    for bloque in _bloques(doc):
        if isinstance(bloque, Table):
            filas = []
            for fila in bloque.rows:
                filas.append([c.text.strip() for c in fila.cells])
            if not filas:
                continue

            n_cols = max(len(f) for f in filas)
            filas = [f + [""] * (n_cols - len(f)) for f in filas]

            # Ancho objetivo por columna, repartiendo el ancho disponible y
            # respetando un mínimo legible. Las celdas largas se envuelven en
            # varias líneas en lugar de truncarse, para no perder información.
            disponible = ANCHO - 3 * (n_cols - 1)
            largos = [max(len(f[i]) for f in filas) for i in range(n_cols)]
            anchos = []
            for largo in largos:
                anchos.append(max(8, min(largo, max(8, disponible // n_cols))))

            # Reparto del espacio sobrante a las columnas que lo necesiten.
            while sum(anchos) < disponible:
                progreso = False
                for i, largo in enumerate(largos):
                    if anchos[i] < largo and sum(anchos) < disponible:
                        anchos[i] += 1
                        progreso = True
                if not progreso:
                    break

            envueltas = [
                [_envolver(celda or " ", ancho=anchos[i]) for i, celda in enumerate(fila)]
                for fila in filas
            ]

            lineas.append("")
            alto = max(len(c) for fila in envueltas for c in fila) if envueltas else 1
            for idx, fila in enumerate(envueltas):
                for sub in range(max(len(c) for c in fila)):
                    parte = [
                        (c[sub] if sub < len(c) else "").ljust(anchos[i])
                        for i, c in enumerate(fila)
                    ]
                    lineas.append(" | ".join(parte).rstrip())
                if idx == 0:
                    lineas.append("-+-".join("-" * a for a in anchos))
            lineas.append("")
            _ = alto
            continue

        texto = bloque.text.strip()
        if not texto:
            continue

        estilo = (bloque.style.name or "").lower()

        if estilo.startswith("heading 1"):
            lineas.append("")
            lineas.append(_regla())
            lineas.append(_centrar(texto.upper()))
            lineas.append(_regla())
            lineas.append("")
        elif estilo.startswith("heading 2"):
            lineas.append("")
            lineas.append(texto)
            lineas.append("-" * min(len(texto), ANCHO))
        elif texto == "Nota." or texto.startswith("Nota."):
            lineas.extend(_envolver(texto, prefijo="    "))
        elif "tabla" in texto.lower() and texto[:6].lower() in ("tabla ", "figura"):
            lineas.append("")
            lineas.append(texto)
        else:
            lineas.extend(_envolver(texto))
            lineas.append("")

    contenido = "\n".join(lineas).rstrip() + "\n\n" + _regla() + "\n" + \
        _centrar("FIN DEL DOCUMENTO") + "\n" + _regla() + "\n"
    SALIDA.write_text(contenido, encoding="utf-8")

    print(f"[INFO] Informe en texto generado: {SALIDA.resolve()}")
    print(f"[INFO] Líneas: {contenido.count(chr(10))}")


def _envolver(texto: str, prefijo: str = "", ancho: int = ANCHO) -> list[str]:
    palabras = texto.split()
    lineas: list[str] = []
    actual = prefijo
    disponible = ancho - len(prefijo)
    for palabra in palabras:
        if len(actual) - len(prefijo) + len(palabra) + (1 if actual.strip() else 0) > disponible:
            lineas.append(actual.rstrip())
            actual = prefijo + palabra + " "
        else:
            actual += palabra + " "
    if actual.strip():
        lineas.append(actual.rstrip())
    return lineas


if __name__ == "__main__":
    exportar()
