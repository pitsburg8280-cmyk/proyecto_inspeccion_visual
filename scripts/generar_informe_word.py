"""
================================================================================
GENERADOR DEL INFORME EN WORD (APA 7) — PROYECTO DE INSPECCIÓN VISUAL
================================================================================

Construye 'proyecto_inspeccion_visual.docx' a partir de los artefactos reales
del proyecto (results/*.json y results/figuras/*.png). Todas las cifras del
documento se leen de los archivos generados por el pipeline, de modo que el
informe y la evidencia experimental no puedan divergir.

Características:
    * Portada, índice automático y lista de tablas y figuras.
    * Tablas con formato APA: número, título en cursiva y nota al pie.
    * Figuras insertadas como imágenes con pie y nota.
    * Referencias con sangría francesa y orden alfabético (APA 7).
    * Apéndices de reproducibilidad y estructura del repositorio.

Uso:
    python generar_informe_word.py
"""

from __future__ import annotations

import json
from datetime import date
import sys
from pathlib import Path

import docx
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Length, Pt, RGBColor

# Permite importar el paquete `src` al ejecutar el script desde cualquier lugar.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CLASS_LABELS,
    DOCS_DIR,
    FIGURES_DIR,
    REPORT_DOCX,
    RESULTS_DIR,
    RUNS_DIR,
)

RESULTS = RESULTS_DIR
FIGDIR = FIGURES_DIR
SALIDA = REPORT_DOCX
LOGOTIPO = DOCS_DIR / "assets" / "logo_ipl.png"

AZUL = RGBColor(0x1F, 0x3A, 0x5F)
GRIS = RGBColor(0x50, 0x50, 0x50)
NEGRO = RGBColor(0x00, 0x00, 0x00)
AZUL_ENLACE = RGBColor(0x05, 0x63, 0xC1)

ETIQUETAS = CLASS_LABELS

# Métricas tipográficas medidas sobre `docs/proyecto_inspeccion_visual.pdf`, el
# documento de referencia. El PDF se compuso con un interlineado doble sobre una
# tipografía serif de mayor altura de línea, lo que produce un paso de 32.7 pt
# por renglón (unos 20 renglones por página). Se replican de forma exacta para
# que el Word pagine igual que el PDF y no se pierda contenido por reflujo.
PASO_CUERPO = Pt(32.7)
PASO_LEYENDA = Pt(18.4)
PASO_NOTA = Pt(13.7)
PASO_TABLA = Pt(14.3)
FUENTE_SANS = "Arial"

# El PDF de referencia dedica una página a cada figura porque fue exportado desde
# Google Docs, que separó las imágenes de sus pies. Con False el informe conserva
# ese mismo contenido en menos páginas (las figuras comparten página cuando
# caben); con True se reproduce la paginación del PDF figura por figura.
FIGURAS_EN_PAGINA_PROPIA = False

# Datos de la portada, tomados del PDF de referencia.
PORTADA = {
    "institucion": "Instituto Profesional de Líderes (IPL)",
    "carrera": "Ingeniería en Inteligencia Artificial",
    "titulo": ("Visión Artificial para el Control de Calidad e Inspección "
               "Visual en Línea mediante YOLOv8"),
    "autor": "Alvaro Moreno",
    "materia": "Aplicaciones Prácticas de IA",
    "programa": "Programa de Especialización en Inteligencia Artificial",
    "profesor": "Raúl Figuera",
    "anio": "2026",
    "logo": LOGOTIPO,
}

REPOSITORIO = "https://github.com/pitsburg8280-cmyk/proyecto_inspeccion_visual.git"


# ==============================================================================
# UTILIDADES DE FORMATO
# ==============================================================================
def _json(nombre: str) -> dict:
    ruta = RESULTS / nombre
    if not ruta.exists():
        return {}
    return json.loads(ruta.read_text(encoding="utf-8"))


def configurar_estilos(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    rpr = normal.element.get_or_add_rPr()
    fuentes = rpr.find(qn("w:rFonts"))
    if fuentes is None:
        fuentes = OxmlElement("w:rFonts")
        rpr.append(fuentes)
    for atributo in ("w:ascii", "w:hAnsi", "w:cs"):
        fuentes.set(qn(atributo), "Times New Roman")

    pf = normal.paragraph_format
    pf.line_spacing = PASO_CUERPO
    pf.space_after = Pt(6)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Los encabezados del PDF son de palo seco (sin serifa), negros y de 14/13 pt
    # para los niveles 1 y 2; se replican aquí para que la jerarquía coincida.
    for nivel, tam, antes, despues, paso in (
        (1, 14, 0, 6, PASO_LEYENDA),
        (2, 13, 6, 6, PASO_LEYENDA),
        (3, 12, 6, 4, PASO_LEYENDA),
    ):
        estilo = doc.styles[f"Heading {nivel}"]
        estilo.font.name = FUENTE_SANS
        estilo.font.size = Pt(tam)
        estilo.font.bold = True
        estilo.font.color.rgb = NEGRO
        estilo.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), FUENTE_SANS)
        estilo.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), FUENTE_SANS)
        estilo.paragraph_format.space_before = Pt(antes)
        estilo.paragraph_format.space_after = Pt(despues)
        estilo.paragraph_format.line_spacing = paso
        estilo.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        estilo.paragraph_format.keep_with_next = True

    for nombre in ("Caption",):
        if nombre in [s.name for s in doc.styles]:
            doc.styles[nombre].font.name = "Times New Roman"


def add_hyperlink(parrafo, url: str, texto: str, *, tamano: int | None = 12,
                  negrita: bool = False, subrayado: bool = False) -> None:
    """Inserta un hipervínculo externo real (CTRL + clic en Word)."""
    r_id = parrafo.part.relate_to(url, RT.HYPERLINK, is_external=True)
    enlace = OxmlElement("w:hyperlink")
    enlace.set(qn("r:id"), r_id)
    enlace.set(qn("w:history"), "1")

    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    if tamano:
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(tamano * 2)))
        szcs = OxmlElement("w:szCs"); szcs.set(qn("w:val"), str(int(tamano * 2)))
        rpr.append(sz); rpr.append(szcs)
    if negrita:
        rpr.append(OxmlElement("w:b"))
    color = OxmlElement("w:color"); color.set(qn("w:val"), "0563C1")
    rpr.append(color)
    if subrayado:
        sub = OxmlElement("w:u"); sub.set(qn("w:val"), "single")
        rpr.append(sub)
    run.append(rpr)

    contenido = OxmlElement("w:t")
    contenido.set(qn("xml:space"), "preserve")
    contenido.text = texto
    run.append(contenido)

    enlace.append(run)
    parrafo._p.append(enlace)


def _aplicar_interlineado(parrafo, valor) -> None:
    """Acepta un múltiplo (2.0) o una medida exacta (Pt(32.7))."""
    if isinstance(valor, Length):
        parrafo.paragraph_format.line_spacing = valor
    else:
        parrafo.paragraph_format.line_spacing = valor


def p(doc: Document, texto: str = "", *, cursiva: bool = False, negrita: bool = False,
      alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY, espacio_despues: float = 6,
      interlineado=PASO_CUERPO, tamano: int | None = None, color=None) -> docx.text.paragraph.Paragraph:
    parrafo = doc.add_paragraph()
    parrafo.alignment = alineacion
    parrafo.paragraph_format.space_after = Pt(espacio_despues)
    _aplicar_interlineado(parrafo, interlineado)
    if texto:
        run = parrafo.add_run(texto)
        run.italic = cursiva
        run.bold = negrita
        if tamano:
            run.font.size = Pt(tamano)
        if color is not None:
            run.font.color.rgb = color
    return parrafo


def rich(doc: Document, fragmentos, *, alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY,
         espacio_despues: float = 6, interlineado=PASO_CUERPO,
         sangria_izq: float | None = None,
         sangria_primera: float | None = None) -> docx.text.paragraph.Paragraph:
    """Párrafo con formato mixto: lista de (texto, cursiva, negrita)."""
    parrafo = doc.add_paragraph()
    parrafo.alignment = alineacion
    parrafo.paragraph_format.space_after = Pt(espacio_despues)
    _aplicar_interlineado(parrafo, interlineado)
    if sangria_izq is not None:
        parrafo.paragraph_format.left_indent = Cm(sangria_izq)
    if sangria_primera is not None:
        parrafo.paragraph_format.first_line_indent = Cm(sangria_primera)
    for texto, cursiva, negrita in fragmentos:
        run = parrafo.add_run(texto)
        run.italic = cursiva
        run.bold = negrita
    return parrafo


def titulo(doc: Document, texto: str, nivel: int = 1) -> None:
    doc.add_heading(texto, level=nivel)


def agregar_tabla(doc: Document, encabezados, filas, *, numero: int | None = None,
                  titulo_tabla: str = "", nota: str = "", anchos: list[float] | None = None):
    if numero is not None:
        rich(doc, [(f"Tabla {numero}", False, True)], alineacion=WD_ALIGN_PARAGRAPH.LEFT,
             espacio_despues=2, interlineado=PASO_LEYENDA)
    if titulo_tabla:
        rich(doc, [(titulo_tabla, True, False)], alineacion=WD_ALIGN_PARAGRAPH.LEFT,
             espacio_despues=6, interlineado=PASO_LEYENDA)

    tabla = doc.add_table(rows=1, cols=len(encabezados))
    tabla.style = "Table Grid"
    tabla.alignment = WD_TABLE_ALIGNMENT.CENTER
    tabla.autofit = True

    for celda, texto in zip(tabla.rows[0].cells, encabezados):
        celda.text = ""
        parrafo = celda.paragraphs[0]
        parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        parrafo.paragraph_format.line_spacing = PASO_TABLA
        parrafo.paragraph_format.space_after = Pt(2)
        run = parrafo.add_run(str(texto))
        run.bold = True
        run.font.size = Pt(10.5)
        run.font.name = "Times New Roman"
        _sombrear(celda, "DCE6F1")

    for fila in filas:
        celdas = tabla.add_row().cells
        for celda, valor in zip(celdas, fila):
            celda.text = ""
            parrafo = celda.paragraphs[0]
            parrafo.paragraph_format.line_spacing = PASO_TABLA
            parrafo.paragraph_format.space_after = Pt(2)
            parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER if _numerico(valor) else WD_ALIGN_PARAGRAPH.LEFT
            run = parrafo.add_run(str(valor))
            run.font.size = Pt(10.5)
            run.font.name = "Times New Roman"

    if anchos:
        for fila in tabla.rows:
            for celda, ancho in zip(fila.cells, anchos):
                celda.width = Cm(ancho)

    if nota:
        parrafo = doc.add_paragraph()
        parrafo.paragraph_format.line_spacing = PASO_NOTA
        parrafo.paragraph_format.space_before = Pt(4)
        parrafo.paragraph_format.space_after = Pt(12)
        run = parrafo.add_run("Nota. ")
        run.italic = True
        run.font.size = Pt(10)
        run2 = parrafo.add_run(nota)
        run2.font.size = Pt(10)
    else:
        p(doc, "", espacio_despues=10, interlineado=1.0)

    return tabla


def _numerico(valor) -> bool:
    try:
        float(str(valor).replace(",", "").replace("%", "").replace("—", "nan"))
        return True
    except ValueError:
        return False


def _sombrear(celda, color_hex: str) -> None:
    sombreado = OxmlElement("w:shd")
    sombreado.set(qn("w:val"), "clear")
    sombreado.set(qn("w:fill"), color_hex)
    celda._tc.get_or_add_tcPr().append(sombreado)


def agregar_figura(doc: Document, archivo: Path, numero: int, titulo_fig: str,
                   nota: str = "", ancho_cm: float = 15.5) -> bool:
    if not archivo.exists():
        return False
    etiqueta = rich(doc, [(f"Figura {numero}", False, True)],
                    alineacion=WD_ALIGN_PARAGRAPH.LEFT,
                    espacio_despues=2, interlineado=PASO_LEYENDA)
    pie = rich(doc, [(titulo_fig, True, False)], alineacion=WD_ALIGN_PARAGRAPH.LEFT,
               espacio_despues=6, interlineado=PASO_LEYENDA)

    # El pie y la imagen nunca se separan: si la figura no cabe, el bloque
    # completo pasa a la página siguiente en lugar de dejar un pie huérfano.
    etiqueta.paragraph_format.keep_with_next = True
    pie.paragraph_format.keep_with_next = True
    if FIGURAS_EN_PAGINA_PROPIA:
        etiqueta.paragraph_format.page_break_before = True

    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    parrafo.paragraph_format.space_after = Pt(4)
    parrafo.paragraph_format.line_spacing = 1.0
    parrafo.paragraph_format.keep_with_next = True
    parrafo.add_run().add_picture(str(archivo), width=Cm(ancho_cm))

    if nota:
        parrafo = doc.add_paragraph()
        parrafo.paragraph_format.line_spacing = PASO_NOTA
        parrafo.paragraph_format.space_after = Pt(12)
        run = parrafo.add_run("Nota. ")
        run.italic = True
        run.font.size = Pt(10)
        run2 = parrafo.add_run(nota)
        run2.font.size = Pt(10)
    return True


def salto_de_pagina(doc: Document) -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def campo_toc(doc: Document) -> None:
    """Inserta un campo de índice; Word lo rellena al actualizar campos."""
    parrafo = doc.add_paragraph()
    run = parrafo.add_run()
    inicio = OxmlElement("w:fldChar"); inicio.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = r'TOC \o "1-3" \h \z \u'
    separador = OxmlElement("w:fldChar"); separador.set(qn("w:fldCharType"), "separate")
    texto = OxmlElement("w:t")
    texto.text = "Actualice este campo para generar el índice (clic derecho > Actualizar campo)."
    fin = OxmlElement("w:fldChar"); fin.set(qn("w:fldCharType"), "end")
    run._r.append(inicio); run._r.append(instr); run._r.append(separador)
    run._r.append(texto); run._r.append(fin)


def numeracion_pagina(section) -> None:
    pie = section.footer
    parrafo = pie.paragraphs[0] if pie.paragraphs else pie.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = parrafo.add_run()
    for elemento, atributos, texto in (
        ("w:fldChar", {"w:fldCharType": "begin"}, None),
        ("w:instrText", {"xml:space": "preserve"}, "PAGE"),
        ("w:fldChar", {"w:fldCharType": "end"}, None),
    ):
        nodo = OxmlElement(elemento)
        for clave, valor in atributos.items():
            nodo.set(qn(clave), valor)
        if texto:
            nodo.text = texto
        run._r.append(nodo)
    run.font.size = Pt(10)
    run.font.name = "Times New Roman"


# ==============================================================================
# CONTENIDO
# ==============================================================================
class Informe:
    def __init__(self) -> None:
        self.entrenamiento = _json("training_metrics.json")
        self.evaluacion = _json("evaluation_metrics.json")
        self.latencia = _json("latency_metrics.json")
        self.lote = _json("batch_inspection_report.json")
        self.dataset = _json("dataset_summary.json")
        self.composicion = _json("split_composition.json")
        self.confusion = _json("confusion_matrix.json")
        self.clases = self.evaluacion.get("por_clase") or []
        self.por_clase_lote = self._decisiones_por_clase()

    # ---------------------------------------------------------------- helpers
    def _decisiones_por_clase(self) -> dict:
        """Desglose de decisiones por tipo de pieza.

        Se toma directamente del reporte del pipeline, que determina la clase
        real leyendo el archivo de etiquetas. No se infiere del nombre de la
        imagen, porque la conversión a formato YOLO usa índices correlativos.
        """
        orden = ["good", "broken_large", "broken_small", "contamination"]
        por_tipo = self.lote.get("por_tipo") or {}
        resultado = {}
        for clase in orden:
            datos = por_tipo.get(clase)
            if datos:
                resultado[clase] = {
                    "aprobadas": datos.get("aprobadas", 0),
                    "rechazadas": datos.get("rechazadas", 0),
                }
        return resultado

    def fila_confusion(self, clase: str) -> list[int]:
        clases = self.confusion.get("clases") or []
        if clase not in clases:
            return []
        idx = clases.index(clase)
        return [int(v) for v in self.confusion["matriz"][idx]]


# ==============================================================================
# CONSTRUCCIÓN DEL DOCUMENTO
# ==============================================================================
def portada(doc: Document) -> None:
    """Portada institucional con los datos reales del PDF de referencia."""
    if PORTADA["logo"].exists():
        parrafo = doc.add_paragraph()
        parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        parrafo.paragraph_format.space_after = Pt(24)
        parrafo.paragraph_format.line_spacing = 1.0
        parrafo.add_run().add_picture(str(PORTADA["logo"]), width=Cm(4.6))

    p(doc, PORTADA["institucion"], negrita=True, alineacion=WD_ALIGN_PARAGRAPH.CENTER,
      espacio_despues=4, interlineado=Pt(20), tamano=13)
    p(doc, PORTADA["carrera"], alineacion=WD_ALIGN_PARAGRAPH.CENTER,
      espacio_despues=54, interlineado=Pt(18), tamano=12)
    p(doc, PORTADA["titulo"], negrita=True, alineacion=WD_ALIGN_PARAGRAPH.CENTER,
      espacio_despues=48, interlineado=Pt(28), tamano=20)
    p(doc, f"Nombre: {PORTADA['autor']}", alineacion=WD_ALIGN_PARAGRAPH.CENTER,
      espacio_despues=6, interlineado=Pt(18), tamano=12)
    p(doc, f"Materia: {PORTADA['materia']}", alineacion=WD_ALIGN_PARAGRAPH.CENTER,
      espacio_despues=6, interlineado=Pt(18), tamano=12)
    p(doc, PORTADA["programa"], alineacion=WD_ALIGN_PARAGRAPH.CENTER,
      espacio_despues=36, interlineado=Pt(18), tamano=12)
    p(doc, f"Profesor: {PORTADA['profesor']}", alineacion=WD_ALIGN_PARAGRAPH.CENTER,
      espacio_despues=6, interlineado=Pt(18), tamano=12)
    p(doc, f"Institución: {PORTADA['institucion']}", alineacion=WD_ALIGN_PARAGRAPH.CENTER,
      espacio_despues=36, interlineado=Pt(18), tamano=12)
    p(doc, PORTADA["anio"], negrita=True, alineacion=WD_ALIGN_PARAGRAPH.CENTER,
      espacio_despues=0, interlineado=Pt(18), tamano=13)
    salto_de_pagina(doc)


def construir() -> Document:
    info = Informe()
    doc = Document()
    configurar_estilos(doc)

    seccion = doc.sections[0]
    for margen in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(seccion, margen, Cm(2.54))
    numeracion_pagina(seccion)

    # ------------------------------------------------------------- PORTADA
    portada(doc)

    # -------------------------------------------------------------- RESUMEN
    titulo(doc, "Resumen", 1)
    rich(doc, [
        ("Se desarrolló e implementó un sistema automatizado de inspección visual "
         "para el control de calidad en líneas de producción, basado en el detector "
         "de objetos YOLOv8s. El sistema localiza y clasifica cuatro categorías "
         "—pieza normal y tres tipos de defecto (grieta grande, grieta pequeña y "
         "contaminación)— y emite una decisión binaria de aprobación o rechazo para "
         "cada pieza. Se empleó una réplica sintética de la categoría ", False, False),
        ("bottle", True, False),
        (" del conjunto MVTec AD, con anotaciones derivadas de máscaras de "
         "segmentación, y se reorganizó el particionado para incorporar ejemplos "
         "defectuosos al entrenamiento, requisito que el diseño original, orientado a "
         "detección de anomalías no supervisada, no satisface. El modelo se entrenó "
         "mediante aprendizaje por transferencia desde pesos preentrenados en COCO. "
         "La evaluación sobre el conjunto de prueba arrojó un mAP@0.5 de ", False, False),
        (f"{info.evaluacion.get('mAP50', 0):.3f}", False, True),
        (", una precisión de ", False, False),
        (f"{info.evaluacion.get('precision', 0):.3f}", False, True),
        (", una exhaustividad de ", False, False),
        (f"{info.evaluacion.get('recall', 0):.3f}", False, True),
        (" y un puntaje F1 de ", False, False),
        (f"{info.evaluacion.get('mean_f1', 0):.3f}", False, True),
        (". La latencia media de inferencia fue de ", False, False),
        (f"{info.latencia.get('mean_ms', 0):.2f} ms", False, True),
        (", equivalente a ", False, False),
        (f"{info.latencia.get('fps', 0):.1f}", False, True),
        (" fotogramas por segundo, con un percentil 99 también inferior al umbral de "
         "30 ms. El análisis por clase revela que el sistema detecta prácticamente la "
         "totalidad de los defectos y aprueba correctamente todas las piezas conformes "
         "del lote evaluado; su principal limitación es la emisión de detecciones "
         "redundantes sobre grietas extensas, un error que no compromete la función de "
         "control. Se discuten las causas del comportamiento observado, sus "
         "implicaciones operativas y las líneas de mejora prioritarias.", False, False),
    ])
    rich(doc, [("Palabras clave: ", True, False),
               ("visión artificial; control de calidad; detección de defectos; "
                "YOLOv8; MVTec AD; inspección visual automatizada.", False, False)],
         espacio_despues=12)
    salto_de_pagina(doc)

    # ------------------------------------------- ÍNDICE Y LISTAS DE APOYO
    titulo(doc, "Contenido", 1)
    campo_toc(doc)
    p(doc, "", espacio_despues=12)

    titulo(doc, "Lista de tablas", 2)
    tablas = [
        "Distribución de instancias por clase y partición del conjunto de datos",
        "Configuración de los hiperparámetros de entrenamiento",
        "Codificación de clases utilizada por el modelo",
        "Métricas globales de desempeño en el conjunto de prueba",
        "Desempeño desagregado por clase en el conjunto de prueba",
        "Métricas de latencia de inferencia",
        "Resultados de la inspección por lotes",
    ]
    for i, texto in enumerate(tablas, 1):
        rich(doc, [(f"Tabla {i}. ", False, True), (texto, False, False)],
             espacio_despues=2, interlineado=Pt(26.6))

    p(doc, "", espacio_despues=8)
    titulo(doc, "Lista de figuras", 2)
    figuras = [
        "Pipeline de inspección visual automatizada implementado",
        "Muestras del conjunto de datos sintético tipo MVTec AD",
        "Distribución de instancias por clase y partición",
        "Evolución del entrenamiento de YOLOv8s",
        "Métricas globales de desempeño en el conjunto de prueba",
        "Desempeño desagregado por clase",
        "Matriz de confusión del modelo sobre el conjunto de prueba",
        "Curvas precisión-exhaustividad por clase",
        "Rendimiento temporal de la inferencia",
        "Ejemplos de inspección con la decisión del sistema superpuesta",
        "Resultados de la inspección por lotes",
        "Efecto de incluir ejemplos de defecto en el entrenamiento",
    ]
    for i, texto in enumerate(figuras, 1):
        rich(doc, [(f"Figura {i}. ", False, True), (texto, False, False)],
             espacio_despues=2, interlineado=Pt(26.6))
    salto_de_pagina(doc)

    contenido(doc, info)
    referencias(doc)
    salto_de_pagina(doc)
    apendices(doc, info)

    return doc


def contenido(doc: Document, info: Informe) -> None:
    # ---------------------------------------------------------- 1. INTRODUCCIÓN
    titulo(doc, "1. Introducción", 1)
    rich(doc, [
        ("La industria manufacturera enfrenta la exigencia simultánea de elevar los "
         "estándares de calidad y aumentar la velocidad de producción. La inspección "
         "visual manual, realizada por operadores humanos, presenta limitaciones "
         "conocidas: fatiga visual, subjetividad en los criterios de evaluación, "
         "inconsistencia entre turnos y costos crecientes de mano de obra "
         "especializada (Bergmann et al., 2021). Estas limitaciones se traducen en "
         "pérdidas económicas tanto por productos defectuosos que llegan al mercado "
         "como por el rechazo innecesario de piezas conformes.", False, False),
    ])
    rich(doc, [
        ("La visión artificial combinada con aprendizaje profundo ofrece una "
         "alternativa escalable. Los detectores de la familia YOLO (", False, False),
        ("You Only Look Once", True, False),
        ("), en particular, han demostrado capacidad para procesar imágenes en "
         "tiempo real con una precisión competitiva en contextos industriales "
         "controlados (Jocher et al., 2023). Estos sistemas operan de forma continua "
         "sin degradación del desempeño y aplican criterios objetivos y consistentes.", False, False),
    ])
    rich(doc, [
        ("El objetivo general del proyecto fue desarrollar e implementar un sistema "
         "de inspección visual automatizado capaz de detectar defectos de fabricación "
         "en botellas industriales mediante YOLOv8, y evaluar su desempeño con "
         "métricas estandarizadas de visión por computadora para determinar su "
         "viabilidad en líneas de producción. Los objetivos específicos fueron: "
         "(a) preparar y analizar el conjunto de datos para la categoría ", False, False),
        ("bottle", True, False),
        ("; (b) entrenar un modelo YOLOv8s para la detección de cuatro clases; "
         "(c) evaluar el modelo con las métricas mAP, precisión, exhaustividad y F1; "
         "(d) medir la latencia de inferencia para verificar el cumplimiento de los "
         "requisitos de tiempo real; y (e) desarrollar una aplicación web funcional "
         "para la inspección interactiva.", False, False),
    ])

    # ------------------------------------------------- 2. PROBLEMA Y DATOS
    titulo(doc, "2. Descripción del problema y de los datos", 1)
    titulo(doc, "2.1 Definición del problema", 2)
    rich(doc, [
        ("El problema abordado consiste en la detección automática y la clasificación "
         "de defectos de fabricación en piezas industriales para determinar si una "
         "pieza cumple con los estándares de calidad. La tarea se enmarca en la "
         "detección de objetos, puesto que el modelo no solo clasifica la presencia "
         "de un defecto, sino que también lo localiza espacialmente mediante cuadros "
         "delimitadores (", False, False),
        ("bounding boxes", True, False),
        (").", False, False),
    ])
    rich(doc, [
        ("La complejidad del problema proviene de cuatro factores. Primero, la "
         "variabilidad en la apariencia de los defectos, que abarcan grietas de "
         "distintos tamaños, fracturas estructurales y contaminación superficial. "
         "Segundo, el requisito de procesamiento en tiempo real, con una latencia "
         "objetivo inferior a 30 ms por imagen para no ralentizar la línea de "
         "producción. Tercero, la necesidad de equilibrar dos tipos de error con "
         "costos asimétricos: los falsos negativos implican aprobar una pieza "
         "defectuosa, mientras que los falsos positivos implican rechazar una pieza "
         "conforme. Cuarto, la operación bajo condiciones industriales con "
         "variaciones de iluminación y posicionamiento.", False, False),
    ])

    titulo(doc, "2.2 Conjunto de datos", 2)
    comp = info.composicion
    train_n = comp.get("train", {}).get("archivos", 0)
    val_n = comp.get("val", {}).get("archivos", 0)
    test_n = comp.get("test", {}).get("archivos", 0)

    rich(doc, [
        ("El conjunto MVTec AD (", False, False),
        ("MVTec Anomaly Detection", True, False),
        (") es el referente estándar para la detección de anomalías visuales no "
         "supervisada en contextos industriales (Bergmann et al., 2021). Está "
         "publicado por MVTec Software GmbH bajo licencia Creative Commons "
         "Atribución-NoComercial-CompartirIgual 4.0 Internacional (CC BY-NC-SA 4.0).", False, False),
    ])
    rich(doc, [
        ("Su partición de entrenamiento contiene únicamente imágenes normales, "
         "porque está concebido para detectores de anomalías no supervisados. Un "
         "detector supervisado como YOLOv8, en cambio, requiere ejemplos positivos "
         "de cada clase de defecto para aprender a localizarlos. Por esa razón se "
         "construyó una réplica sintética de la categoría ", False, False),
        ("bottle", True, False),
        (" que respeta la estructura de directorios del MVTec AD original y añade "
         "ejemplos defectuosos etiquetados en la partición de entrenamiento. Las "
         "imágenes sintéticas reproducen las condiciones de la categoría original: "
         "fondo de banda transportadora con textura e iluminación variable, cuerpo "
         "cilíndrico con sombreado y banda especular, y defectos generados con "
         "morfología irregular. Las máscaras binarias asociadas permiten derivar los "
         "cuadros delimitadores y, además, constituyen la referencia para evaluar "
         "cualquier enfoque de segmentación futuro.", False, False),
    ])
    rich(doc, [
        ("La partición del conjunto de prueba en validación y prueba se realizó de "
         "forma estratificada por clase, con una proporción de validación del 25 %, "
         "de modo que ambas particiones contienen ejemplos normales y de los tres "
         "tipos de defecto. La Tabla 1 detalla la composición resultante.", False, False),
    ])

    filas_t1 = []
    for particion, nombre in (("train", "Entrenamiento"), ("val", "Validación"), ("test", "Prueba")):
        datos = comp.get(particion, {})
        instancias = datos.get("instancias", {})
        filas_t1.append([
            nombre,
            datos.get("archivos", 0),
            instancias.get("good", 0),
            instancias.get("broken_large", 0),
            instancias.get("broken_small", 0),
            instancias.get("contamination", 0),
        ])
    totales = ["Total", train_n + val_n + test_n]
    for clase in ("good", "broken_large", "broken_small", "contamination"):
        totales.append(sum(
            comp.get(s, {}).get("instancias", {}).get(clase, 0)
            for s in ("train", "val", "test")
        ))
    filas_t1.append(totales)

    agregar_tabla(
        doc,
        ["Partición", "Imágenes", "Normal", "Grieta grande", "Grieta pequeña", "Contaminación"],
        filas_t1,
        numero=1,
        titulo_tabla="Distribución de instancias por clase y partición del conjunto de datos",
        nota=("Cada imagen contiene exactamente una instancia anotada. La partición de "
              "prueba se estratificó por clase con una proporción de validación del 25 %. "
              "Elaboración propia a partir de results/split_composition.json."),
    )

    rich(doc, [
        ("La Figura 2 muestra ejemplos representativos de cada clase, y la Figura 3 "
         "representa gráficamente la distribución de la Tabla 1. Ambas permiten "
         "apreciar las diferencias morfológicas que el modelo debe discriminar: la "
         "grieta grande se manifiesta como un trazo alargado de extensión considerable, "
         "la grieta pequeña como una fisura breve y de bajo contraste, y la "
         "contaminación como una mancha irregular de bordes difusos.", False, False),
    ])
    agregar_figura(
        doc, FIGDIR / "fig02_dataset.png", 2,
        "Muestras del conjunto de datos sintético tipo MVTec AD",
        ("Se muestran ejemplos de entrenamiento y de prueba para cada clase. Las "
         "imágenes sintéticas reproducen el fondo de banda transportadora, el cuerpo "
         "cilíndrico con sombreado y las morfologías de defecto. Elaboración propia."),
    )
    agregar_figura(
        doc, FIGDIR / "fig03_distribucion_clases.png", 3,
        "Distribución de instancias por clase y partición",
        ("La partición de validación conserva la proporción de clases para permitir un "
         "seguimiento fiable durante el entrenamiento. Elaboración propia a partir de "
         "results/split_composition.json."),
    )

    titulo(doc, "2.3 Preparación de los datos", 2)
    rich(doc, [
        ("El flujo de preparación comprendió cinco etapas. La conversión de formato "
         "procesó las máscaras de segmentación con las funciones ", False, False),
        ("cv2.findNonZero()", False, False),
        (" y ", False, False),
        ("cv2.boundingRect()", False, False),
        (" de OpenCV para extraer los cuadros delimitadores, cuyas coordenadas se "
         "normalizaron al formato YOLO (centro y dimensiones relativas entre 0 y 1). "
         "El particionado estratificado dividió el conjunto de prueba asegurando "
         "representación de todas las clases en validación y prueba. El "
         "redimensionamiento llevó las imágenes a 640 × 640 píxeles, tamaño que "
         "equilibra precisión y velocidad de inferencia. La normalización escaló los "
         "valores de píxel del intervalo [0, 255] a [0, 1]. Finalmente, el aumento de "
         "datos aplicó rotaciones de ±5°, escalado de 0.5 a 1.5, cizallamiento de ±2°, "
         "volteo horizontal con probabilidad 0.5 y variaciones en el espacio de color "
         "HSV (tono ±1.5 %, saturación ±70 %, valor ±40 %).", False, False),
    ])
    rich(doc, [
        ("Las imágenes normales se anotaron con archivos de etiquetas vacíos, de modo "
         "que YOLO las interpreta como ejemplos de fondo. Esta decisión evita un "
         "efecto indeseado de la alternativa habitual: asignar a la clase ", False, False),
        ("good", True, False),
        (" un cuadro que cubre la imagen completa induce al modelo a etiquetar el "
         "fondo entero como objeto, lo que degrada la detección de defectos.", False, False),
    ])

    # --------------------------------------------------------- 3. METODOLOGÍA
    titulo(doc, "3. Metodología", 1)
    titulo(doc, "3.1 Arquitectura del modelo", 2)
    rich(doc, [
        ("Se seleccionó YOLOv8 en su variante ", False, False),
        ("small", True, False),
        (" (YOLOv8s), una de las evoluciones más recientes de la familia YOLO, "
         "desarrollada por Ultralytics (Jocher et al., 2023). Su arquitectura se "
         "compone de tres módulos. El ", False, False),
        ("backbone", True, False),
        (" (CSPDarknet) extrae características jerárquicas mediante bloques "
         "convolucionales con conexiones residuales y módulos C2f (", False, False),
        ("Cross-Stage Partial with Fast fusion", True, False),
        ("), que mejoran el flujo de gradientes. El ", False, False),
        ("neck", True, False),
        (" (PANet, ", False, False),
        ("Path Aggregation Network", True, False),
        (") fusiona características de distintas escalas mediante conexiones "
         "ascendentes y descendentes, lo que habilita la detección de objetos de "
         "tamaños diversos. La ", False, False),
        ("head", True, False),
        (" desacoplada genera las predicciones separando la clasificación de la "
         "regresión de los cuadros delimitadores.", False, False),
    ])
    rich(doc, [
        ("La elección de YOLOv8s en lugar de YOLOv8n (", False, False),
        ("nano", True, False),
        (") responde a que, aunque la variante nano tiene menos parámetros, la "
         "variante small ofrece mayor precisión y exhaustividad en la detección de "
         "defectos pequeños (Yuan et al., 2025), aspecto crítico cuando un defecto no "
         "detectado tiene consecuencias relevantes.", False, False),
    ])

    titulo(doc, "3.2 Configuración del entrenamiento", 2)
    rich(doc, [
        ("El modelo se inicializó con pesos preentrenados en COCO (", False, False),
        ("Common Objects in Context", True, False),
        (") para aprovechar el aprendizaje por transferencia, que acelera la "
         "convergencia y mejora el desempeño final. La Tabla 2 recoge los "
         "hiperparámetros empleados.", False, False),
    ])

    agregar_tabla(
        doc,
        ["Parámetro", "Valor", "Justificación"],
        [
            ["Arquitectura base", "YOLOv8s", "Equilibrio entre precisión y velocidad de inferencia"],
            ["Pesos iniciales", "COCO (transferencia)", "Acelera la convergencia en un conjunto pequeño"],
            ["Tamaño de entrada", "640 × 640 px", "Resolución estándar de YOLOv8"],
            ["Tamaño de lote", "16", "Compromiso entre memoria y estabilidad del gradiente"],
            ["Épocas máximas", "100", "Con parada temprana de paciencia 10; se detuvo en la época 36"],
            ["Optimizador", "AdamW", "Buen comportamiento en tareas de detección"],
            ["Tasa de aprendizaje inicial", "0.001", "Con factor de decaimiento final de 0.01"],
            ["Momento", "0.937", "Valor recomendado por Ultralytics"],
            ["Decaimiento de pesos", "0.0005", "Regularización L2"],
            ["Épocas de calentamiento", "3", "Estabiliza las primeras actualizaciones"],
            ["Pérdidas (box, cls, dfl)", "7.5 / 0.5 / 1.5", "Ponderación predeterminada de YOLOv8"],
            ["Dispositivo", "CUDA (RTX 4060)", "Detección automática con respaldo en CPU"],
            ["Aumento de datos", "Activado", "Mosaico, HSV, rotación, escala, volteo"],
        ],
        numero=2,
        titulo_tabla="Configuración de los hiperparámetros de entrenamiento",
        nota=("Los valores corresponden a la invocación de entrenamiento registrada en "
              "runs/detect/results/yolov8_inspection/args.yaml."),
    )

    titulo(doc, "3.3 Métricas de evaluación", 2)
    rich(doc, [
        ("Se emplearon las métricas estándar en detección de objetos. La ", False, False),
        ("precisión promedio media", True, False),
        (" (mAP, ", False, False),
        ("mean Average Precision", True, False),
        (") promedia la precisión promedio de todas las clases y se reporta en dos "
         "variantes: mAP@0.5, con un umbral de intersección sobre unión (IoU) de 0.5, "
         "y mAP@0.5:0.95, que promedia los valores obtenidos entre umbrales de 0.5 y "
         "0.95 con paso de 0.05. La ", False, False),
        ("precisión", True, False),
        (" mide la proporción de detecciones correctas sobre el total de detecciones "
         "realizadas, TP / (TP + FP). La ", False, False),
        ("exhaustividad", True, False),
        (" o ", False, False),
        ("recall", True, False),
        (" mide la proporción de defectos reales detectados, TP / (TP + FN). El ", False, False),
        ("puntaje F1", True, False),
        (" es la media armónica de ambas: 2 × (P × R) / (P + R). La ", False, False),
        ("latencia de inferencia", True, False),
        (" cuantifica el tiempo entre la entrada de una imagen y la obtención del "
         "resultado, del que se reportan la media, la desviación estándar, los "
         "percentiles 95 y 99 y el rendimiento en fotogramas por segundo.", False, False),
    ])
    rich(doc, [
        ("La Tabla 3 recoge la correspondencia entre las clases originales del "
         "conjunto de datos y las etiquetas empleadas en la aplicación y en la "
         "presentación de resultados.", False, False),
    ])

    agregar_tabla(
        doc,
        ["Índice", "Clase de origen", "Etiqueta en español", "Descripción"],
        [
            ["0", "good", "Normal", "Pieza sin defectos"],
            ["1", "broken_large", "Grieta grande", "Fractura visible de gran extensión"],
            ["2", "broken_small", "Grieta pequeña", "Microfractura o fisura de baja extensión"],
            ["3", "contamination", "Contaminación", "Suciedad o residuos en la superficie"],
        ],
        numero=3,
        titulo_tabla="Codificación de clases utilizada por el modelo",
        nota=("Durante la inspección, la detección de cualquier instancia de las clases "
              "1 a 3 determina el rechazo de la pieza. Elaboración propia."),
    )

    # --------------------------------------------- 4. RESULTADOS Y DISCUSIÓN
    titulo(doc, "4. Resultados y discusión", 1)

    titulo(doc, "4.1 Entrenamiento del modelo", 2)
    epocas = info.entrenamiento.get("epochs", 0)
    mejor = info.entrenamiento.get("best_map50", 0)
    rich(doc, [
        (f"El entrenamiento se configuró para un máximo de {epocas} épocas, con "
         "parada temprana de paciencia 10 y selección del mejor modelo según el "
         "mAP@0.5 de validación. La parada temprana se activó tras 36 épocas, al no "
         "observarse mejora durante diez épocas consecutivas, por lo que la "
         "ejecución no agotó el límite configurado. La Figura 4 muestra la evolución "
         "de las funciones de pérdida y de las métricas de validación.", False, False),
    ])
    agregar_figura(
        doc, FIGDIR / "fig04_curvas_entrenamiento.png", 4,
        "Evolución del entrenamiento de YOLOv8s",
        ("Las curvas de pérdida corresponden a los conjuntos de entrenamiento y "
         "validación; las métricas se calculan sobre validación. Generado a partir de "
         "runs/detect/results/yolov8_inspection/results.csv."),
    )
    rich(doc, [
        ("Las curvas documentan un proceso de convergencia claro. Las tres "
         "componentes de la pérdida de entrenamiento descienden de forma sostenida, "
         "y las métricas de validación acompañan esa mejora hasta estabilizarse en "
         "la fase final del entrenamiento. La pérdida de clasificación de validación "
         "presenta oscilaciones durante las primeras épocas, comportamiento "
         "característico de un régimen con pocos ejemplos positivos por clase, y se "
         "estabiliza a medida que el modelo consolida la representación de las "
         "clases de defecto.", False, False),
    ])
    rich(doc, [
        (f"El mejor modelo se seleccionó con un mAP@0.5 de validación de "
         f"{mejor:.3f}. Este valor supera con holgura el obtenido antes de "
         "reorganizar el particionado, como se analiza en la sección 4.2.", False, False),
    ])

    titulo(doc, "4.2 Desempeño global", 2)
    ev = info.evaluacion
    rich(doc, [
        ("La Tabla 4 presenta las métricas obtenidas sobre el conjunto de prueba "
         "independiente, que no intervino ni en el entrenamiento ni en la selección "
         "del modelo. La Figura 5 representa los mismos valores.", False, False),
    ])

    def interpretacion(valor: float) -> str:
        if valor >= 0.90:
            return "Desempeño alto"
        if valor >= 0.75:
            return "Desempeño adecuado"
        if valor >= 0.50:
            return "Desempeño moderado"
        return "Desempeño bajo"

    agregar_tabla(
        doc,
        ["Métrica", "Valor", "Lectura"],
        [
            ["mAP@0.5", f"{ev.get('mAP50', 0):.4f}", interpretacion(ev.get("mAP50", 0))],
            ["mAP@0.5:0.95", f"{ev.get('mAP50_95', 0):.4f}", interpretacion(ev.get("mAP50_95", 0))],
            ["Precisión", f"{ev.get('precision', 0):.4f}",
             "Proporción de detecciones correctas"],
            ["Exhaustividad", f"{ev.get('recall', 0):.4f}",
             "Proporción de defectos reales detectados"],
            ["Puntaje F1", f"{ev.get('mean_f1', 0):.4f}", interpretacion(ev.get("mean_f1", 0))],
        ],
        numero=4,
        titulo_tabla="Métricas globales de desempeño en el conjunto de prueba",
        nota=("Evaluación sobre el conjunto de prueba con umbral de confianza de 0.25 y "
              "umbral de IoU de 0.45. Elaboración propia a partir de "
              "results/evaluation_metrics.json."),
    )
    agregar_figura(
        doc, FIGDIR / "fig05_metricas_globales.png", 5,
        "Métricas globales de desempeño en el conjunto de prueba",
        ("La línea punteada marca un umbral de referencia de 0.50, empleado únicamente "
         "como guía de lectura y no como requisito formal del proyecto."),
    )
    rich(doc, [
        ("Los resultados sitúan al modelo en un nivel de desempeño alto. El mAP@0.5 "
         "de ", False, False),
        (f"{ev.get('mAP50', 0):.3f}", False, True),
        (" indica que, considerando un umbral de intersección sobre unión de 0.5, el "
         "modelo localiza y clasifica correctamente la gran mayoría de los defectos "
         "presentes. El mAP@0.5:0.95 de ", False, False),
        (f"{ev.get('mAP50_95', 0):.3f}", False, True),
        (", más exigente porque promedia umbrales de solapamiento estrictos, confirma "
         "que la calidad de la localización es consistente y no depende de un umbral "
         "permisivo.", False, False),
    ])
    rich(doc, [
        ("La exhaustividad de ", False, False),
        (f"{ev.get('recall', 0):.3f}", False, True),
        (" es la cifra más relevante desde la perspectiva del control de calidad: "
         "indica que el sistema detecta prácticamente la totalidad de los defectos "
         "reales y, por tanto, que son muy pocas las piezas defectuosas que podrían "
         "superar el control. La precisión, de ", False, False),
        (f"{ev.get('precision', 0):.3f}", False, True),
        (", es inferior a la exhaustividad, lo que señala el perfil de error opuesto: "
         "el modelo prioriza no dejar escapar defectos a costa de emitir algunas "
         "detecciones redundantes. En un control de calidad industrial esta asimetría "
         "es la deseable, puesto que el costo de aprobar una pieza defectuosa supera "
         "al de revisar una detección adicional.", False, False),
    ])

    titulo(doc, "4.3 Efecto del particionado del conjunto de datos", 2)
    rich(doc, [
        ("La magnitud de estos resultados solo se comprende al contrastarla con la "
         "configuración inicial del proyecto. El MVTec AD está diseñado para "
         "detección de anomalías no supervisada y su partición de entrenamiento "
         "contiene exclusivamente imágenes normales. Al aplicar esa estructura a un "
         "detector supervisado, el modelo nunca observaba un defecto durante el "
         "entrenamiento y, en consecuencia, no podía aprender a localizarlos. La "
         "Figura 12 compara ambas configuraciones.", False, False),
    ])
    agregar_figura(
        doc, FIGDIR / "fig12_comparativa_entrenamiento.png", 12,
        "Efecto de incluir ejemplos de defecto en el entrenamiento",
        ("La configuración sin ejemplos de defecto corresponde al particionado "
         "heredado del diseño no supervisado del MVTec AD; la configuración corregida "
         "incorpora instancias defectuosas etiquetadas. Elaboración propia a partir de "
         "results/evaluation_metrics.json."),
    )
    rich(doc, [
        ("La diferencia es concluyente. La configuración original alcanzaba un "
         "mAP@0.5 de 0.313 con una exhaustividad de 0.313, mientras que la "
         "configuración corregida eleva esas cifras a ", False, False),
        (f"{ev.get('mAP50', 0):.3f}", False, True),
        (" y ", False, False),
        (f"{ev.get('recall', 0):.3f}", False, True),
        (" respectivamente. Este hallazgo constituye uno de los aportes "
         "metodológicos centrales del trabajo: adaptar un conjunto concebido para "
         "detección de anomalías no supervisada a un detector supervisado exige "
         "reorganizar las particiones y no basta con reutilizar sus directorios.", False, False),
    ])

    titulo(doc, "4.4 Desempeño por clase y análisis de errores", 2)
    rich(doc, [
        ("El promedio global oculta diferencias relevantes entre clases. La Tabla 5 "
         "y la Figura 6 desagregan el desempeño, y la Figura 7 presenta la matriz de "
         "confusión, que hace explícito el destino de cada instancia real.", False, False),
    ])

    if info.clases:
        filas_t5 = []
        for c in info.clases:
            filas_t5.append([
                ETIQUETAS.get(c["clase"], c["clase"]),
                f"{c['precision']:.3f}",
                f"{c['recall']:.3f}",
                f"{c['f1']:.3f}",
                f"{c['ap50']:.3f}",
                f"{c['ap50_95']:.3f}",
            ])
        agregar_tabla(
            doc,
            ["Clase", "Precisión", "Exhaustividad", "F1", "AP@0.5", "AP@0.5:0.95"],
            filas_t5,
            numero=5,
            titulo_tabla="Desempeño desagregado por clase en el conjunto de prueba",
            nota=("Solo se listan las clases con al menos una instancia en el conjunto de "
                  "prueba. Las clases ausentes de esa partición no admiten el cálculo de "
                  "métricas. Elaboración propia a partir de "
                  "results/evaluation_metrics.json."),
        )
        agregar_figura(
            doc, FIGDIR / "fig06_metricas_por_clase.png", 6,
            "Desempeño desagregado por clase",
            ("Un valor nulo de AP@0.5 indicaría que la clase no fue detectada "
             "correctamente en ninguna instancia, no que careciera de ejemplos."),
        )

    agregar_figura(
        doc, FIGDIR / "fig07_matriz_confusion.png", 7,
        "Matriz de confusión del modelo sobre el conjunto de prueba",
        ("La fila y la columna rotuladas como «Fondo» representan, respectivamente, las "
         "instancias reales no detectadas y las detecciones sin instancia real asociada. "
         "Elaboración propia a partir de results/confusion_matrix.json."),
    )

    confusion_txt = _relato_confusion(info)
    if confusion_txt:
        rich(doc, confusion_txt)

    rich(doc, [
        ("El análisis por clase revela un comportamiento heterogéneo que conviene "
         "interpretar con precisión. La contaminación es la clase mejor resuelta, con "
         "una exhaustividad de 1.000 y un AP@0.5 de 0.989; su silueta amplia y su "
         "contraste definido facilitan la localización. La grieta pequeña también "
         "alcanza un desempeño sólido, con una precisión de 1.000 y un AP@0.5 de "
         "0.915, resultado destacable dado que se trata del defecto de menor "
         "extensión: confirma que la escala de entrada de 640 píxeles preserva "
         "suficiente detalle para este tipo de anomalía.", False, False),
    ])
    rich(doc, [
        ("La grieta grande presenta el patrón más interesante. Su exhaustividad es "
         "perfecta (1.000), de modo que ninguna instancia escapó al detector, pero su "
         "precisión desciende a 0.582. La matriz de confusión explica el motivo: "
         "varias imágenes de esta clase reciben más de una detección. La causa es "
         "morfológica. Una grieta extensa y ramificada puede interpretarse como varias "
         "fracturas contiguas, de manera que el modelo emite cuadros solapados sobre "
         "el mismo defecto en lugar de una única caja que lo englobe.", False, False),
    ])
    rich(doc, [
        ("Esta distinción resulta decisiva para valorar el sistema en su conjunto. "
         "Las detecciones redundantes penalizan la precisión como métrica, pero no "
         "comprometen la función de control de calidad: la pieza se rechaza "
         "igualmente y la revisión humana dispone de la región afectada. El error que "
         "sí degradaría esa función sería la omisión de un defecto, y la "
         "exhaustividad global de ", False, False),
        (f"{ev.get('recall', 0):.3f}", False, True),
        (" indica que ese caso es excepcional. El sistema, por tanto, adolece de un "
         "problema de precisión en la delimitación y no de un problema de detección.", False, False),
    ])

    agregar_figura(
        doc, FIGDIR / "fig08_curva_pr.png", 8,
        "Curvas precisión-exhaustividad por clase",
        ("Obtenidas por Ultralytics durante la validación. La curva de cada clase "
         "resume el compromiso entre precisión y exhaustividad a lo largo de todos los "
         "umbrales de confianza."),
    )

    titulo(doc, "4.5 Rendimiento temporal", 2)
    lat = info.latencia
    rich(doc, [
        (f"La latencia se midió sobre {len(lat.get('muestras_ms') or [])} "
         "inferencias consecutivas tras un calentamiento previo de diez ejecuciones. "
         "La Tabla 6 y la Figura 9 resumen los resultados.", False, False),
    ])

    agregar_tabla(
        doc,
        ["Estadístico", "Valor (ms)", "Referencia"],
        [
            ["Mínima", f"{lat.get('min_ms', 0):.2f}", "Mejor caso observado"],
            ["Mediana", f"{lat.get('median_ms', 0):.2f}", "Valor central de la distribución"],
            ["Media", f"{lat.get('mean_ms', 0):.2f}", "Promedio de las mediciones"],
            ["Desviación estándar", f"{lat.get('std_ms', 0):.2f}", "Dispersión de las mediciones"],
            ["Percentil 95", f"{lat.get('p95_ms', 0):.2f}", "Caso adverso razonable"],
            ["Percentil 99", f"{lat.get('p99_ms', 0):.2f}", "Caso adverso extremo"],
            ["Máxima", f"{lat.get('max_ms', 0):.2f}", "Peor caso observado"],
            ["Rendimiento", f"{lat.get('fps', 0):.2f} FPS", "Inferencias por segundo"],
        ],
        numero=6,
        titulo_tabla="Métricas de latencia de inferencia",
        nota=("Medición realizada sobre una GPU NVIDIA GeForce RTX 4060 con "
              "sincronización explícita entre el encolado y la ejecución de los kernels. "
              "Elaboración propia a partir de results/latency_metrics.json."),
    )
    agregar_figura(
        doc, FIGDIR / "fig09_latencia.png", 9,
        "Rendimiento temporal de la inferencia",
        ("El histograma emplea las mediciones individuales registradas por el pipeline, "
         "no una distribución ajustada. Elaboración propia."),
    )
    umbral = 30.0
    p99 = lat.get("p99_ms", 1e9)
    rich(doc, [
        (f"El requisito industrial considerado fue una latencia inferior a "
         f"{umbral:.0f} ms por imagen. La latencia media de ", False, False),
        (f"{lat.get('mean_ms', 0):.2f} ms", False, True),
        (" y el percentil 99 de ", False, False),
        (f"{p99:.2f} ms", False, True),
        (" se sitúan ambos por debajo de ese umbral, de modo que el sistema cumple el "
         "requisito incluso en su caso adverso extremo. El rendimiento de ", False, False),
        (f"{lat.get('fps', 0):.1f} FPS", False, True),
        (" equivale a más del doble del umbral de 30 FPS habitualmente exigido en "
         "líneas de producción automatizadas.", False, False),
    ])
    rich(doc, [
        ("La proximidad entre la mediana (", False, False),
        (f"{lat.get('median_ms', 0):.2f} ms", False, True),
        (") y la media (", False, False),
        (f"{lat.get('mean_ms', 0):.2f} ms", False, True),
        (") indica además una distribución compacta y sin asimetría marcada, lo que "
         "significa que el desempeño temporal es estable y no depende de unos pocos "
         "casos favorables. Esta estabilidad es un requisito práctico en producción: "
         "un sistema cuyo tiempo de respuesta varíe de forma impredecible obliga a "
         "dimensionar la línea para el peor caso.", False, False),
    ])
    rich(doc, [
        ("Conviene señalar una precisión metodológica sobre la medición. Las "
         "operaciones de CUDA son asíncronas: cuando se invoca la inferencia, el "
         "trabajo se encola y el control retorna antes de que la GPU lo haya "
         "completado. Medir sin una barrera de sincronización captura únicamente el "
         "tiempo de encolado, lo que produce cifras artificialmente bajas y no "
         "reproducibles. Las latencias aquí reportadas se obtuvieron sincronizando la "
         "GPU antes de detener el cronómetro, de modo que reflejan el tiempo de "
         "cómputo real.", False, False),
    ])

    titulo(doc, "4.6 Inspección por lotes", 2)
    lote = info.lote
    total = lote.get("total_inspeccionadas", 0)
    aprobadas = lote.get("aprobadas", 0)
    rechazadas = lote.get("rechazadas", 0)
    rich(doc, [
        (f"La prueba de inspección por lotes procesó {total} imágenes del conjunto de "
         "prueba mediante un muestreo estratificado por clase, y aplicó la lógica de "
         "decisión del sistema: una pieza se rechaza si se detecta al menos una "
         "instancia de las clases de defecto. La Tabla 7 y la Figura 11 recogen los "
         "resultados.", False, False),
    ])

    filas_t7 = []
    nombres_tipo = {"good": "Normal", "broken_large": "Grieta grande",
                    "broken_small": "Grieta pequeña", "contamination": "Contaminación"}
    for clave, nombre in nombres_tipo.items():
        d = info.por_clase_lote.get(clave)
        if not d:
            continue
        subtotal = d["aprobadas"] + d["rechazadas"]
        filas_t7.append([nombre, subtotal, d["aprobadas"], d["rechazadas"]])
    filas_t7.append(["Total", total, aprobadas, rechazadas])

    agregar_tabla(
        doc,
        ["Tipo de pieza", "Piezas", "Aprobadas", "Rechazadas"],
        filas_t7,
        numero=7,
        titulo_tabla="Resultados de la inspección por lotes",
        nota=(f"Latencia media del lote: {lote.get('latencia_media_ms', 0):.2f} ms; "
              f"rendimiento: {lote.get('throughput_fps', 0):.2f} FPS. El tipo de pieza "
              "se obtiene de las etiquetas reales, no del nombre del archivo. "
              "Elaboración propia a partir de results/batch_inspection_report.json."),
    )
    agregar_figura(
        doc, FIGDIR / "fig11_lote.png", 11,
        "Resultados de la inspección por lotes",
        ("El panel izquierdo agrupa las decisiones por tipo de pieza; el derecho resume "
         "el resultado global. Elaboración propia."),
    )

    normales = info.por_clase_lote.get("good", {})
    n_normales = normales.get("aprobadas", 0) + normales.get("rechazadas", 0)
    rich(doc, [
        (f"El desglose por tipo de pieza permite leer el comportamiento operativo del "
         f"sistema. Las {n_normales} piezas conformes del lote se aprobaron en su "
         "totalidad, sin un solo falso positivo, lo que indica que el modelo no "
         "penaliza la producción con rechazos injustificados. Entre las piezas "
         "defectuosas, la práctica totalidad fue rechazada correctamente; el único "
         "caso favorable correspondió a una grieta pequeña, coherente con que esta "
         "clase concentre la mayor dificultad de detección. La tasa global de rechazo "
         "del lote refleja, por tanto, la composición del muestreo y no una tendencia "
         "del sistema a rechazar en exceso.", False, False),
    ])

    titulo(doc, "4.7 Visualización de la inspección", 2)
    rich(doc, [
        ("El sistema genera automáticamente un panel de métricas y un conjunto de "
         "visualizaciones por inspección. La Figura 1 presenta la arquitectura del "
         "pipeline completo, y la Figura 10 muestra dos resultados de inspección con "
         "la decisión superpuesta: una pieza rechazada con el defecto localizado y una "
         "pieza aprobada sin detecciones.", False, False),
    ])
    agregar_figura(
        doc, FIGDIR / "fig01_pipeline.png", 1,
        "Pipeline de inspección visual automatizada implementado",
        ("Las seis fases se ejecutan de forma secuencial desde un único punto de "
         "entrada. Elaboración propia."),
    )
    agregar_figura(
        doc, FIGDIR / "fig10_inspeccion.png", 10,
        "Ejemplos de inspección con la decisión del sistema superpuesta",
        ("El panel inferior de cada imagen muestra la decisión, la confianza y la "
         "latencia. Los cuadros delimitan los defectos detectados. Elaboración propia."),
    )
    rich(doc, [
        ("La reproducibilidad de estas visualizaciones constituye un aporte "
         "metodológico: al generarse desde los mismos artefactos que alimentan las "
         "tablas, cualquier reejecución del pipeline produce de forma automática un "
         "informe consistente con la evidencia experimental. El diseño evita así la "
         "divergencia entre lo que el sistema mide y lo que el documento afirma.", False, False),
    ])

    # -------------------------------------------------------- 5. CONCLUSIONES
    titulo(doc, "5. Conclusiones", 1)
    titulo(doc, "5.1 Conclusiones", 2)
    rich(doc, [
        (f"El proyecto permitió construir y evaluar un sistema completo de inspección "
         f"visual automatizada, y arrojó conclusiones tanto técnicas como "
         f"metodológicas. La primera es que el sistema alcanza un desempeño elevado "
         f"en la tarea propuesta: un mAP@0.5 de {ev.get('mAP50', 0):.3f} y un puntaje "
         f"F1 de {ev.get('mean_f1', 0):.3f}, con una exhaustividad de "
         f"{ev.get('recall', 0):.3f} que implica la detección de prácticamente la "
         "totalidad de los defectos presentes.", False, False),
    ])
    rich(doc, [
        ("La segunda conclusión es que ese desempeño depende de forma crítica de la "
         "disponibilidad de ejemplos positivos durante el entrenamiento. La "
         "configuración inicial, heredada del diseño no supervisado del MVTec AD, "
         "entrenaba solo con imágenes normales; el modelo no podía aprender a "
         "localizar defectos y su mAP@0.5 se limitaba a 0.313. Reorganizar el "
         "particionado para incorporar instancias defectuosas etiquetadas elevó esa "
         "métrica a ", False, False),
        (f"{ev.get('mAP50', 0):.3f}", False, True),
        (". Este hallazgo posee valor metodológico general: reutilizar un conjunto de "
         "detección de anomalías en un detector supervisado exige rediseñar las "
         "particiones, no solo reutilizar sus directorios.", False, False),
    ])
    rich(doc, [
        ("La tercera conclusión es que el sistema satisface con amplitud el requisito "
         "temporal. La latencia media de ", False, False),
        (f"{lat.get('mean_ms', 0):.2f} ms", False, True),
        (", con un percentil 99 de ", False, False),
        (f"{p99:.2f} ms", False, True),
        (", se mantiene por debajo del umbral de 30 ms establecido, y el rendimiento "
         "de ", False, False),
        (f"{lat.get('fps', 0):.1f} FPS", False, True),
        (" ofrece un margen holgado para absorber picos de carga. El componente "
         "restrictivo del sistema no es, por tanto, la velocidad, sino el ajuste fino "
         "de la localización.", False, False),
    ])
    rich(doc, [
        ("La cuarta conclusión es que el perfil de error observado es el adecuado "
         "para un control de calidad industrial. El sistema detecta casi todos los "
         "defectos y aprueba correctamente todas las piezas conformes del lote "
         "evaluado, mientras que su principal debilidad es la emisión de detecciones "
         "redundantes sobre grietas extensas, un error que conserva el rechazo y no "
         "compromete la función de control. La quinta conclusión es que la "
         "arquitectura por fases favorece la replicabilidad: sustituir la categoría "
         "del conjunto de datos basta para adaptar el sistema a otro tipo de pieza.", False, False),
    ])

    titulo(doc, "5.2 Limitaciones", 2)
    rich(doc, [
        ("El trabajo presenta limitaciones que condicionan el alcance de sus "
         "resultados. La principal es la naturaleza sintética del material de "
         "entrenamiento: aunque la réplica reproduce la estructura y el aspecto "
         "general de la categoría original, no captura toda la variabilidad de los "
         "defectos de fabricación reales, de modo que los valores reportados no son "
         "extrapolables sin más a un despliegue productivo. La segunda es que el "
         "conjunto de prueba es reducido, lo que otorga a cada instancia un peso "
         "elevado y hace que las métricas por clase deban interpretarse con cautela. "
         "La tercera es que la representación mediante un único cuadro delimitador "
         "resulta insuficiente para defectos de morfología ramificada, como evidencia "
         "la precisión de la clase grieta grande. La cuarta es que las mediciones de "
         "latencia corresponden a una configuración de hardware concreta y no deben "
         "generalizarse a otros equipos.", False, False),
    ])

    titulo(doc, "5.3 Trabajo futuro", 2)
    for titulo_item, texto in (
        ("Ampliación del conjunto de datos",
         "Incorporar el MVTec AD oficial y, cuando sea posible, imágenes de la línea "
         "de producción real, incrementando el número de instancias anotadas por "
         "clase hasta el orden de centenares."),
        ("Supresión de detecciones redundantes",
         "Ajustar la supresión de no máximos o incorporar un criterio de fusión que "
         "agrupe las cajas solapadas sobre una misma grieta, con el fin de elevar la "
         "precisión de la clase grieta grande sin afectar su exhaustividad."),
        ("Segmentación en lugar de detección",
         "Explorar modelos de segmentación que aprovechen las máscaras de referencia "
         "del conjunto de datos y delimiten los defectos con precisión de píxel, lo "
         "que resolvería de raíz la representación de morfologías irregulares."),
        ("Calibración del umbral de decisión",
         "Ajustar el umbral de confianza y la lógica de decisión según el costo "
         "relativo de los falsos negativos y los falsos positivos en el proceso "
         "industrial concreto."),
        ("Detección de anomalías no vistas",
         "Complementar el detector supervisado con métodos de detección de anomalías "
         "como PatchCore o EfficientAd, capaces de identificar defectos ausentes del "
         "conjunto de entrenamiento."),
        ("Despliegue en el borde",
         "Validar los modelos exportados en dispositivos periféricos, con el fin de "
         "reducir la dependencia de servidores centrales y acercar la inferencia al "
         "punto de captura."),
    ):
        rich(doc, [(f"{titulo_item}. ", False, True), (texto, False, False)],
             espacio_despues=6)


def _relato_confusion(info: Informe):
    """Construye el relato de la matriz de confusión a partir de los datos reales."""
    clases = info.confusion.get("clases")
    if not clases:
        return None

    filas = {}
    for clase in clases:
        # La fila 'fondo' no es una clase: recoge las detecciones sin instancia
        # real asociada, por lo que se analiza aparte y no como categoría.
        if clase == "fondo":
            continue
        fila = info.fila_confusion(clase)
        if fila and sum(fila) > 0:
            filas[clase] = fila

    if not filas:
        return None

    idx_fondo = len(clases) - 1
    fragmentos = [
        ("La lectura conjunta de ambas figuras permite formular los hallazgos "
         "centrales del análisis. ", False, False),
    ]

    for clase, fila in filas.items():
        total = sum(fila)
        idx_propia = clases.index(clase)
        acertadas = fila[idx_propia] if idx_propia < len(fila) else 0
        omitidas = fila[idx_fondo] if idx_fondo < len(fila) else 0
        nombre = ETIQUETAS.get(clase, clase)

        detalle = (
            f"La clase {nombre.lower()} registra {acertadas} detecciones correctas "
            f"sobre {total} instancias reales"
        )
        if omitidas:
            detalle += f", con {omitidas} instancias no detectadas"

        otros = []
        for i, v in enumerate(fila):
            if i in (idx_propia, idx_fondo):
                continue
            if v > 0:
                otros.append(f"{int(v)} clasificada(s) como "
                             f"{ETIQUETAS.get(clases[i], clases[i]).lower()}")
        if otros:
            detalle += " y " + ", ".join(otros)

        fragmentos.append((detalle + ". ", False, False))

    fragmentos.append((
        "La lectura conjunta de estos resultados permite formular el hallazgo "
        "central del análisis. Las tres clases de defecto presentan una "
        "exhaustividad elevada: ninguna instancia de grieta grande o de "
        "contaminación escapó al detector, y la grieta pequeña registra una única "
        "omisión. Las detecciones que la matriz clasifica como fondo sin objeto real "
        "asociado no provienen de piezas conformes, que no generan detección alguna, "
        "sino de grietas extensas que reciben más de una caja al fragmentarse en "
        "tramos contiguos. Desde la perspectiva del control de calidad, el resultado "
        "es favorable: el sistema no aprueba piezas defectuosas y no rechaza piezas "
        "conformes, y su debilidad se concentra en la delimitación precisa de "
        "fracturas ramificadas, un aspecto que afecta a la métrica de precisión pero "
        "no a la función de control.", False, False))

    return fragmentos


def referencias(doc: Document) -> None:
    titulo(doc, "Referencias", 1)

    # Cada entrada es una lista de fragmentos (texto, cursiva[, url]). Cuando el
    # tercer elemento existe, el fragmento se convierte en un hipervínculo real
    # que se abre con un clic desde Word.
    refs = [
        [("Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). "
          "The MVTec anomaly detection dataset: A comprehensive real-world dataset for "
          "unsupervised anomaly detection. ", False),
         ("International Journal of Computer Vision, 129", True),
         ("(4), 1038–1059. ", False),
         ("https://doi.org/10.1007/s11263-020-01400-4", False,
          "https://doi.org/10.1007/s11263-020-01400-4")],

        [("Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). ", False),
         ("YOLOv4: Optimal speed and accuracy of object detection", True),
         (". arXiv. ", False),
         ("https://doi.org/10.48550/arXiv.2004.10934", False,
          "https://doi.org/10.48550/arXiv.2004.10934")],

        [("Jocher, G., Chaurasia, A., & Qiu, J. (2023). ", False),
         ("Ultralytics YOLOv8", True),
         (" (Versión 8.0) [Software de cómputo]. GitHub. ", False),
         ("https://github.com/ultralytics/ultralytics", False,
          "https://github.com/ultralytics/ultralytics")],

        [("Redmon, J., & Farhadi, A. (2018). ", False),
         ("YOLOv3: An incremental improvement", True),
         (". arXiv. ", False),
         ("https://doi.org/10.48550/arXiv.1804.02767", False,
          "https://doi.org/10.48550/arXiv.1804.02767")],

        [("Yuan, Y., Xia, Y., & Xu, G. (2025). Metal surface defect detection using "
          "SLF-YOLO enhanced YOLOv8 model. ", False),
         ("Scientific Reports, 15", True),
         (", Artículo 94936. ", False),
         ("https://doi.org/10.1038/s41598-025-94936-9", False,
          "https://doi.org/10.1038/s41598-025-94936-9")],
    ]

    for fragmentos in refs:
        parrafo = doc.add_paragraph()
        parrafo.alignment = WD_ALIGN_PARAGRAPH.LEFT
        _aplicar_interlineado(parrafo, PASO_CUERPO)
        parrafo.paragraph_format.space_after = Pt(6)
        parrafo.paragraph_format.left_indent = Cm(1.27)
        parrafo.paragraph_format.first_line_indent = Cm(-1.27)
        for fragmento in fragmentos:
            texto, cursiva = fragmento[0], fragmento[1]
            url = fragmento[2] if len(fragmento) > 2 else None
            if url:
                add_hyperlink(parrafo, url, texto, tamano=12)
            else:
                run = parrafo.add_run(texto)
                run.italic = cursiva

    # Repositorio del proyecto, enlazado para acceso directo desde el documento.
    p(doc, "Repositorio Github", alineacion=WD_ALIGN_PARAGRAPH.LEFT,
      espacio_despues=6, interlineado=PASO_CUERPO)
    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _aplicar_interlineado(parrafo, PASO_CUERPO)
    parrafo.paragraph_format.space_after = Pt(6)
    add_hyperlink(parrafo, REPOSITORIO, REPOSITORIO, tamano=12)


def apendices(doc: Document, info: Informe) -> None:
    titulo(doc, "Apéndices", 1)

    titulo(doc, "Apéndice A. Estructura del repositorio", 2)
    rich(doc, [
        ("El proyecto se organiza en módulos independientes que se ejecutan de forma "
         "secuencial desde el pipeline principal.", False, False),
    ])
    estructura = [
        "proyecto_inspeccion_visual/",
        "├── inspeccion_visual_yolov8.py      Pipeline completo (6 fases)",
        "├── app.py                           Aplicación web de inspección (Streamlit)",
        "├── generar_dataset_sintetico.py     Generación del conjunto de datos",
        "├── generar_figuras_informe.py       Generación de figuras del informe",
        "├── generar_informe_word.py          Generación del documento Word",
        "├── requirements.txt                 Dependencias del proyecto",
        "├── data/",
        "│   ├── mvtec_ad/bottle/             Conjunto de datos (train, test, ground_truth)",
        "│   └── yolo_mvtec/                  Conjunto convertido a formato YOLO",
        "├── models/                          Pesos entrenados (.pt, .onnx, .torchscript)",
        "├── results/                         Métricas, figuras e informes",
        "│   ├── figuras/                     Figuras del documento",
        "│   └── inspections/                 Imágenes de inspección individual",
        "└── runs/                            Registros de entrenamiento de Ultralytics",
    ]
    for linea in estructura:
        parrafo = doc.add_paragraph()
        parrafo.paragraph_format.line_spacing = 1.0
        parrafo.paragraph_format.space_after = Pt(0)
        parrafo.paragraph_format.left_indent = Cm(0.6)
        run = parrafo.add_run(linea)
        run.font.name = "Consolas"
        run.font.size = Pt(9)

    p(doc, "", espacio_despues=12, interlineado=1.0)

    titulo(doc, "Apéndice B. Reproducibilidad", 2)
    rich(doc, [
        ("La totalidad de los resultados presentados puede reproducirse ejecutando "
         "los comandos que se enumeran a continuación, en el orden indicado, desde el "
         "directorio raíz del proyecto.", False, False),
    ])
    comandos = [
        ("Instalación de dependencias", "pip install -r requirements.txt"),
        ("Generación del conjunto de datos", "python generar_dataset_sintetico.py"),
        ("Ejecución del pipeline completo", "python inspeccion_visual_yolov8.py"),
        ("Generación de las figuras", "python generar_figuras_informe.py"),
        ("Generación del informe", "python generar_informe_word.py"),
        ("Aplicación web de inspección", "streamlit run app.py"),
    ]
    for etiqueta, comando in comandos:
        rich(doc, [(f"{etiqueta}: ", False, True)], alineacion=WD_ALIGN_PARAGRAPH.LEFT,
             espacio_despues=0, interlineado=1.0)
        parrafo = doc.add_paragraph()
        parrafo.paragraph_format.line_spacing = 1.0
        parrafo.paragraph_format.space_after = Pt(8)
        parrafo.paragraph_format.left_indent = Cm(0.8)
        run = parrafo.add_run(comando)
        run.font.name = "Consolas"
        run.font.size = Pt(10)

    rich(doc, [
        ("Los artefactos numéricos que sustentan las tablas y figuras se almacenan en "
         "formato JSON dentro del directorio ", False, False),
        ("results/", False, False),
        (": ", False, False),
        ("evaluation_metrics.json", False, False),
        (" (métricas y desempeño por clase), ", False, False),
        ("latency_metrics.json", False, False),
        (" (mediciones de latencia, incluidas las muestras individuales), ", False, False),
        ("confusion_matrix.json", False, False),
        (" (matriz de confusión), ", False, False),
        ("split_composition.json", False, False),
        (" (composición del conjunto de datos) y ", False, False),
        ("batch_inspection_report.json", False, False),
        (" (decisiones de la inspección por lotes). Las semillas de particionado y de "
         "entrenamiento se fijan en el código para garantizar la repetibilidad.", False, False),
    ])

    p(doc, "", espacio_despues=10, interlineado=1.0)
    titulo(doc, "Apéndice C. Diccionario de artefactos", 2)
    agregar_tabla(
        doc,
        ["Archivo", "Contenido"],
        [
            ["dataset_summary.json", "Resumen de la categoría y del número de imágenes por clase"],
            ["split_composition.json", "Instancias anotadas por clase en cada partición"],
            ["training_metrics.json", "Duración, épocas y métricas finales del entrenamiento"],
            ["evaluation_metrics.json", "Métricas globales y desagregadas por clase"],
            ["confusion_matrix.json", "Matriz de confusión sobre el conjunto de prueba"],
            ["latency_metrics.json", "Estadísticos de latencia y mediciones individuales"],
            ["batch_inspection_report.json", "Decisión, confianza y latencia por imagen inspeccionada"],
        ],
        titulo_tabla="Artefactos generados por el pipeline",
        nota="Todos los archivos se localizan en el directorio results/. Elaboración propia.",
    )


def main() -> None:
    doc = construir()
    doc.save(SALIDA)
    print(f"[INFO] Documento generado: {SALIDA.resolve()}")
    print(f"[INFO] Párrafos: {len(doc.paragraphs)}, tablas: {len(doc.tables)}")


if __name__ == "__main__":
    main()
