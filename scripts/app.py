
"""
================================================================================
APLICACIÓN WEB: INSPECTOR DE CALIDAD VISUAL CON IA
================================================================================
Interfaz web para la inspección automática de piezas con YOLOv8.

EJECUCIÓN:
    streamlit run scripts/app.py

La lógica de decisión (umbrales, reglas de aprobación y confianza) reside en
`src/inspection.py`, compartida con el pipeline, de modo que ambas rutas de uso
del sistema no pueden divergir.
================================================================================
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Permite importar el paquete `src` al ejecutar la app desde cualquier lugar.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CLASS_LABELS,
    CLASS_NAMES,
    CONFIG,
    INSPECTIONS_DIR,
    RESULTS_DIR,
    TRAINED_DIR,
    pretrained_weights,
    trained_model,
)
from src.inspection import inspeccionar  # noqa: E402

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
st.set_page_config(
    page_title="Inspector de Calidad Visual - IA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .approved {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .rejected {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ESTADO DE LA SESIÓN
# ==============================================================================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'modelo_entrenado' not in st.session_state:
    st.session_state.modelo_entrenado = False
if 'inspection_history' not in st.session_state:
    st.session_state.inspection_history = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# ==============================================================================
# FUNCIONES
# ==============================================================================

@st.cache_resource
def load_model(model_path: str | None = None):
    """
    Carga el modelo YOLOv8 entrenado.

    Si no existe el modelo entrenado, recurre a los pesos preentrenados como
    respaldo. En ese caso las detecciones no son fiables: el modelo base no
    conoce las clases de defecto del proyecto.
    """
    ruta = Path(model_path) if model_path else trained_model()
    if not ruta.exists():
        return YOLO(str(pretrained_weights())), False
    return YOLO(str(ruta)), True


def inspect_image(image, model, conf_threshold=CONFIG["confidence_threshold"],
                  iou_threshold=CONFIG["iou_threshold"]):
    """
    Realiza la inspección visual de una imagen y devuelve la vista anotada.

    La decisión se delega en `src.inspection.inspeccionar`, la misma función que
    emplea el pipeline, para que los criterios de aprobación sean idénticos en
    ambos caminos de uso.

    Returns:
        result_img: imagen anotada con las cajas y el panel de decisión.
        decision: "APROBADA" o "RECHAZADA".
        confidence: confianza del defecto que motiva el rechazo, o la holgura de
            la aprobación respecto del umbral.
        latency_ms: tiempo de inferencia en milisegundos.
        defects: lista de defectos detectados.
    """
    # Convertir PIL a numpy (BGR, que es lo que espera OpenCV y YOLO)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    resultado = inspeccionar(model, img_bgr, conf=conf_threshold, iou=iou_threshold)

    result_img = img_bgr.copy()
    defects = []

    for defecto in resultado.defectos:
        x1, y1, x2, y2 = (int(v) for v in defecto.bbox)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        etiqueta = CLASS_LABELS.get(defecto.tipo, defecto.tipo)
        label = f"{etiqueta}: {defecto.confianza:.1%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result_img, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 255), -1)
        cv2.putText(result_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        defects.append({
            "tipo": etiqueta,
            "confianza": defecto.confianza,
            "ubicacion": f"({x1},{y1})-({x2},{y2})",
        })

    decision = resultado.decision
    confidence = resultado.confianza if not resultado.aprobada else resultado.margen_aprobacion

    # Panel de decisión
    panel_h = 100
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:, :] = (0, 200, 0) if resultado.aprobada else (0, 0, 200)

    text1 = f"DECISIÓN: {decision}"
    etiqueta_conf = "Confianza" if not resultado.aprobada else "Margen de aprobación"
    text2 = f"{etiqueta_conf}: {confidence:.1%}  |  Latencia: {resultado.latencia_ms:.1f}ms"

    cv2.putText(panel, text1, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(panel, text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    final_img = np.vstack([result_img, panel])
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    return final_rgb, decision, confidence, resultado.latencia_ms, defects

def load_metrics():
    """Carga métricas del modelo si existen."""
    metrics_files = {
        "training": str(RESULTS_DIR / "training_metrics.json"),
        "evaluation": str(RESULTS_DIR / "evaluation_metrics.json"),
        "latency": str(RESULTS_DIR / "latency_metrics.json")
    }
    metrics = {}
    for key, path in metrics_files.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                metrics[key] = json.load(f)
    return metrics if metrics else None

# ==============================================================================
# INTERFAZ PRINCIPAL
# ==============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Inspector de Calidad Visual con IA</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema automatizado de detección de defectos de fabricación '
                'mediante Visión Artificial e Inteligencia Artificial</p>', 
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")

        # Cargar modelo
        model_path = st.text_input("Ruta del modelo", str(trained_model()))

        if st.button("🚀 Cargar Modelo", type="primary"):
            with st.spinner("Cargando modelo YOLOv8..."):
                try:
                    modelo, es_entrenado = load_model(model_path)
                    st.session_state.model = modelo
                    st.session_state.modelo_entrenado = es_entrenado
                    if es_entrenado:
                        st.success("✅ Modelo cargado correctamente")
                    else:
                        st.warning("⚠️ No se encontró el modelo entrenado. Se usan los "
                                   "pesos preentrenados como respaldo, por lo que las "
                                   "detecciones no son fiables.")
                except Exception as e:
                    st.error(f"❌ Error cargando modelo: {e}")

        st.divider()

        # Parámetros de inferencia
        st.subheader("Parámetros de Inferencia")
        conf_threshold = st.slider("Umbral de Confianza", 0.0, 1.0, 0.25, 0.01)
        iou_threshold = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)

        st.divider()

        # Información del proyecto
        st.subheader("📋 Información del Proyecto")
        st.markdown("""
        **Materia:** Aplicaciones Prácticas de IA  
        **Modelo:** YOLOv8 (Small)  
        **Dataset:** MVTec AD  
        **Categoría:** Bottle  
        **Tarea:** Detección de Defectos  
        """)

        st.divider()

        # Métricas del modelo
        if st.button("📊 Cargar Métricas"):
            st.session_state.metrics = load_metrics()

    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Inspección Individual", "📁 Inspección por Lotes", 
                                       "📈 Dashboard de Métricas", "📚 Documentación"])

    # TAB 1: Inspección Individual
    with tab1:
        st.header("Inspección de Pieza Individual")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Cargar Imagen")
            uploaded_file = st.file_uploader(
                "Seleccione una imagen de la pieza a inspeccionar",
                type=["png", "jpg", "jpeg", "bmp"],
                help="Soporta formatos PNG, JPG, JPEG, BMP"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Imagen cargada", use_container_width=True)

                if st.button("🔍 Iniciar Inspección", type="primary", use_container_width=True):
                    if st.session_state.model is None:
                        st.warning("⚠️ Primero cargue el modelo desde el panel lateral")
                    else:
                        with st.spinner("Analizando imagen con IA..."):
                            result_img, decision, confidence, latency, defects = inspect_image(
                                image, st.session_state.model, conf_threshold, iou_threshold
                            )

                            # Guardar en historial
                            st.session_state.inspection_history.append({
                                "imagen": uploaded_file.name,
                                "decision": decision,
                                "confianza": confidence,
                                "latencia_ms": latency,
                                "defectos": len(defects),
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })

                        with col2:
                            st.subheader("Resultado de la Inspección")
                            st.image(result_img, caption="Resultado del análisis", 
                                    use_container_width=True)

                            # Panel de decisión
                            if decision == "APROBADA":
                                st.markdown(f'<div class="approved">✅ {decision}</div>', 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="rejected">❌ {decision}</div>', 
                                          unsafe_allow_html=True)

                            # Métricas
                            mcol1, mcol2, mcol3 = st.columns(3)
                            with mcol1:
                                st.metric("Confianza", f"{confidence:.1%}")
                            with mcol2:
                                st.metric("Latencia", f"{latency:.1f} ms")
                            with mcol3:
                                st.metric("Defectos", len(defects))

                            # Detalles de defectos
                            if defects:
                                st.subheader("🚨 Defectos Detectados")
                                for i, defect in enumerate(defects, 1):
                                    st.error(f"**Defecto {i}:** {defect['tipo']}  "
                                            f"(Confianza: {defect['confianza']:.1%})  "
                                            f"- Ubicación: {defect['ubicacion']}")
                            else:
                                st.success("✨ No se detectaron defectos. La pieza cumple con los estándares de calidad.")

        # Historial de inspecciones
        if st.session_state.inspection_history:
            st.divider()
            st.subheader("📜 Historial de Inspecciones")
            hist_df = pd.DataFrame(st.session_state.inspection_history)
            st.dataframe(hist_df, use_container_width=True)

            if st.button("🗑️ Limpiar Historial"):
                st.session_state.inspection_history = []
                st.rerun()

    # TAB 2: Inspección por Lotes
    with tab2:
        st.header("Inspección por Lotes")

        batch_files = st.file_uploader(
            "Seleccione múltiples imágenes",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if batch_files and st.button("🔍 Inspeccionar Lote", type="primary"):
            if st.session_state.model is None:
                st.warning("⚠️ Primero cargue el modelo")
            else:
                progress_bar = st.progress(0)
                batch_results = []

                for idx, file in enumerate(batch_files):
                    progress_bar.progress((idx + 1) / len(batch_files))

                    image = Image.open(file).convert("RGB")
                    result_img, decision, confidence, latency, defects = inspect_image(
                        image, st.session_state.model, conf_threshold, iou_threshold
                    )

                    batch_results.append({
                        "imagen": file.name,
                        "decision": decision,
                        "confianza": confidence,
                        "latencia_ms": latency,
                        "defectos": len(defects)
                    })

                progress_bar.empty()

                # Resumen
                approved = sum(1 for r in batch_results if r["decision"] == "APROBADA")
                rejected = len(batch_results) - approved
                avg_latency = np.mean([r["latencia_ms"] for r in batch_results])

                st.success(f"✅ Lote procesado: {approved} aprobadas, {rejected} rechazadas")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", len(batch_results))
                col2.metric("Aprobadas", approved, f"{approved/len(batch_results)*100:.0f}%")
                col3.metric("Rechazadas", rejected, f"{rejected/len(batch_results)*100:.0f}%")
                col4.metric("Latencia Media", f"{avg_latency:.1f} ms")

                # Tabla de resultados
                st.subheader("Resultados Detallados")
                results_df = pd.DataFrame(batch_results)
                st.dataframe(results_df, use_container_width=True)

                # Descargar reporte
                report_json = json.dumps(batch_results, indent=2)
                st.download_button(
                    "📥 Descargar Reporte JSON",
                    report_json,
                    file_name=f"reporte_inspeccion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    # TAB 3: Dashboard de Métricas
    with tab3:
        st.header("📈 Dashboard de Métricas del Modelo")

        if st.session_state.metrics:
            metrics = st.session_state.metrics

            col1, col2, col3 = st.columns(3)

            if "evaluation" in metrics:
                eval_m = metrics["evaluation"]
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("mAP@0.5", f"{eval_m.get('mAP50', 0):.3f}")
                    st.metric("mAP@0.5:0.95", f"{eval_m.get('mAP50_95', 0):.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Precisión", f"{eval_m.get('precision', 0):.3f}")
                    st.metric("Recall", f"{eval_m.get('recall', 0):.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("F1-Score", f"{eval_m.get('mean_f1', 0):.3f}")
                    st.metric("mAP", f"{eval_m.get('mAP', 0):.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)

            if "latency" in metrics:
                st.subheader("⚡ Métricas de Latencia")
                lat = metrics["latency"]
                lcol1, lcol2, lcol3, lcol4, lcol5 = st.columns(5)
                lcol1.metric("Media (ms)", f"{lat.get('mean_ms', 0):.1f}")
                lcol2.metric("Mediana (ms)", f"{lat.get('median_ms', 0):.1f}")
                lcol3.metric("P95 (ms)", f"{lat.get('p95_ms', 0):.1f}")
                lcol4.metric("P99 (ms)", f"{lat.get('p99_ms', 0):.1f}")
                lcol5.metric("FPS", f"{lat.get('fps', 0):.1f}")

            if "training" in metrics:
                st.subheader("🎓 Métricas de Entrenamiento")
                train = metrics["training"]
                tcol1, tcol2, tcol3 = st.columns(3)
                tcol1.metric("Épocas", train.get("epochs", "N/A"))
                tcol2.metric("Tiempo (s)", f"{train.get('training_time_seconds', 0):.0f}")
                tcol3.metric("Best mAP50", f"{train.get('best_map50', 0):.3f}")
        else:
            st.info("ℹ️ Haga clic en 'Cargar Métricas' en el panel lateral para visualizar el rendimiento del modelo.")

            # Métricas de ejemplo/demostración
            st.subheader("Métricas de Referencia (Esperadas)")
            st.markdown("""
            Basado en entrenamientos con el dataset MVTec AD (categoría Bottle):

            | Métrica | Valor Esperado |
            |---------|---------------|
            | mAP@0.5 | 0.85 - 0.92 |
            | mAP@0.5:0.95 | 0.65 - 0.75 |
            | Precisión | 0.88 - 0.95 |
            | Recall | 0.82 - 0.90 |
            | F1-Score | 0.85 - 0.92 |
            | Latencia Media | 15 - 25 ms |
            | Throughput | 40 - 65 FPS |
            """)

    # TAB 4: Documentación
    with tab4:
        st.header("📚 Documentación del Sistema")

        st.markdown("""
        ### 🎯 Propósito
        Este sistema implementa un pipeline completo de Visión Artificial para el control de calidad 
        automatizado en líneas de producción industrial, utilizando el modelo YOLOv8 entrenado con 
        el dataset MVTec AD.

        ### 🔧 Arquitectura del Sistema

        ```
        ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
        │  Cámara/      │────▶│  YOLOv8          │────▶│  Decisión       │
        │  Imagen       │     │  Detector        │     │  Aprobación     │
        │  de Entrada   │     │  de Defectos     │     │  o Rechazo      │
        └─────────────────┘     └──────────────────┘     └─────────────────┘
                                       │
                                       ▼
                                ┌──────────────────┐
                                │  Métricas:       │
                                │  Precisión,      │
                                │  Recall, mAP,    │
                                │  Latencia        │
                                └──────────────────┘
        ```

        ### 📋 Tipos de Defectos Detectados

        | Clase | Nombre | Descripción |
        |-------|--------|-------------|
        | 0 | Normal | Pieza sin defectos |
        | 1 | Grieta Grande | Fractura visible significativa |
        | 2 | Grieta Pequeña | Micro-fractura o fisura |
        | 3 | Contaminación | Suciedad o residuos en superficie |

        ### ⚙️ Parámetros Configurables

        - **Umbral de Confianza**: Controla la sensibilidad de detección (0.0 - 1.0)
        - **Umbral IoU**: Controla la superposición mínima para considerar una detección válida
        - **Tamaño de imagen**: 640x640 píxeles (optimizado para YOLOv8)

        ### 📊 Métricas de Evaluación

        El sistema reporta las siguientes métricas estándar de Visión Artificial:
        - **mAP (mean Average Precision)**: Precisión promedio ponderada
        - **Precisión**: Proporción de detecciones correctas
        - **Recall**: Proporción de defectos detectados correctamente
        - **F1-Score**: Media armónica de precisión y recall
        - **Latencia**: Tiempo de procesamiento por imagen (objetivo: <30ms)

        ### 🚀 Despliegue

        El modelo puede exportarse a los siguientes formatos:
        - **ONNX**: Para despliegue en servidores y edge devices
        - **TorchScript**: Para integración con aplicaciones PyTorch
        - **OpenVINO**: Para aceleración en hardware Intel

        ### 📖 Referencias

        - Bergmann, P., et al. (2021). The MVTec Anomaly Detection Dataset. *International Journal of Computer Vision*, 129, 1038-1059.
        - Jocher, G., et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
        """)

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    main()
