# 🔍 Visión Artificial para Control de Calidad e Inspección Visual en Línea

**Detección de defectos en piezas industriales con YOLOv8 y MVTec AD**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-orange)
![mAP@0.5](https://img.shields.io/badge/mAP%400.5-0.926-brightgreen)
![Latencia](https://img.shields.io/badge/latencia-12.5%20ms-yellow)

Sistema de inspección visual automatizada que localiza y clasifica defectos de
fabricación y decide si cada pieza se **APRUEBA** o se **RECHAZA**, con pipeline
completo de datos, entrenamiento, evaluación, reporte y aplicación web.

---

## ⚡ Inicio rápido

```bash
# 1. Entorno y dependencias
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Conjunto de datos (réplica sintética reproducible, no requiere descargas)
python scripts/generar_dataset_sintetico.py

# 3. Pipeline completo: datos -> entrenamiento -> evaluación -> exportación
python scripts/inspeccion_visual_yolov8.py

# 4. Interfaz web de inspección
streamlit run scripts/app.py
```

> 💡 **GPU.** Si dispones de una GPU NVIDIA, instala antes PyTorch con CUDA
> (véase [Instalación](#-instalación)). En CPU el pipeline funciona, pero el
> entrenamiento pasa de minutos a horas.

> 📁 Todos los comandos se ejecutan **desde la raíz del proyecto**. Las rutas
> están ancladas a la raíz mediante `src/config.py`, de modo que también
> funcionan si los invocas desde otro directorio.

---

## 📋 Tabla de contenidos

- [Inicio rápido](#-inicio-rápido)
- [Descripción del proyecto](#-descripción-del-proyecto)
- [Características](#-características)
- [Arquitectura del sistema](#-arquitectura-del-sistema)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Obtención del conjunto de datos](#-obtención-del-conjunto-de-datos)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Reproducibilidad](#-reproducibilidad)
- [Solución de problemas](#-solución-de-problemas)
- [Referencias](#-referencias)
- [Informe académico](#-informe-académico)

---

## 🎯 Descripción del proyecto

Este proyecto implementa un **sistema automatizado de control de calidad visual**
para líneas de producción industrial. Emplea un detector de objetos **YOLOv8**
entrenado sobre una réplica sintética de la categoría *bottle* del conjunto
**MVTec AD**, con el que localiza y clasifica defectos de fabricación y determina
si cada pieza debe aprobarse o rechazarse.

El sistema sustituye la inspección visual manual —lenta, subjetiva y propensa a
errores— por un criterio objetivo, consistente y capaz de operar de forma
continua. El repositorio cubre el ciclo completo: preparación de datos,
entrenamiento, evaluación con métricas estandarizadas, medición de latencia,
inspección por lotes, exportación a formatos de despliegue, aplicación web y
generación automática del informe técnico.

**Problema que resuelve.** La inspección manual presenta tasas de error elevadas
por fatiga visual, subjetividad e inconsistencia entre operadores, lo que genera
pérdidas económicas tanto por productos defectuosos que llegan al mercado como
por el rechazo innecesario de piezas conformes.

> ### ⚠️ Nota metodológica
>
> El MVTec AD está diseñado para **detección de anomalías no supervisada**: su
> partición de entrenamiento contiene **únicamente imágenes normales**. Un
> detector supervisado como YOLOv8 necesita ejemplos positivos de cada defecto
> para aprender a localizarlos. Por eso este proyecto usa una **réplica
> sintética** que conserva la estructura de directorios del original y **añade
> ejemplos defectuosos etiquetados al entrenamiento**.
>
> Esta diferencia no es un detalle: entrenar solo con imágenes normales limita el
> mAP@0.5 a ≈ 0.31 porque el modelo nunca observa un defecto. Incorporar ejemplos
> defectuosos lo eleva a **0.926** (véase [Resultados](#-resultados)).

---

## ✨ Características

- **Detección de cuatro clases:** Normal, Grieta grande, Grieta pequeña y Contaminación.
- **Pipeline reproducible de extremo a extremo** con seis fases automatizadas.
- **Métricas estandarizadas:** mAP@0.5, mAP@0.5:0.95, precisión, exhaustividad y F1, globales y por clase.
- **Matriz de confusión** calculada con emparejamiento explícito por IoU.
- **Análisis de latencia** riguroso: media, desviación estándar, percentiles 95 y 99, y FPS.
- **Inspección individual y por lotes**, con desglose de decisiones por tipo de pieza.
- **Aplicación web** (Streamlit) con inspección interactiva y panel de métricas.
- **Exportación del modelo** a ONNX y TorchScript para despliegue.
- **Informe técnico autogenerado** en Word y texto a partir de los artefactos medidos.
- **Figuras generadas desde los datos**, de modo que el informe no puede divergir de la evidencia.

---

## 🧠 Arquitectura del sistema

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Cámara /       │────▶│  YOLOv8s         │────▶│  Decisión       │
│  imagen de      │     │  Detector de     │     │  APROBADA /     │
│  entrada        │     │  defectos        │     │  RECHAZADA      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                          │
                                ▼                          ▼
                        ┌──────────────────┐      ┌─────────────────┐
                        │  Métricas:       │      │  Panel visual   │
                        │  mAP, precisión, │      │  con cajas y    │
                        │  exhaustividad,  │      │  decisión       │
                        │  latencia        │      └─────────────────┘
                        └──────────────────┘
```

### Fases del pipeline (`scripts/inspeccion_visual_yolov8.py`)

| Fase | Etapa | Descripción |
|------|-------|-------------|
| 1 | Preparación del conjunto de datos | Valida la estructura, inventaría clases y verifica máscaras |
| 2 | Exploración (EDA) | Visualiza muestras y estadísticas de píxeles |
| 3 | Conversión a YOLO y entrenamiento | Genera `data.yaml`, deriva cajas de las máscaras, particiona de forma estratificada y entrena YOLOv8s |
| 4 | Evaluación | Calcula mAP, precisión, exhaustividad, F1, métricas por clase, matriz de confusión y latencia |
| 5 | Inspección por lotes | Analiza un lote estratificado y emite decisiones con desglose por tipo |
| 6 | Exportación | Exporta el modelo a ONNX y TorchScript |

### Decisiones de diseño relevantes

- **Imágenes normales sin anotación.** Las etiquetas de las imágenes normales
  quedan vacías (fondo). Asignarles en su lugar una caja de clase *good* que
  cubra la imagen completa enseña al modelo a delimitar el fondo entero.
- **Particionado estratificado.** Validación y prueba se reparten por clase, de
  modo que ambas contienen ejemplos normales y de los tres defectos.
- **Clase leída de la etiqueta, no del nombre.** La conversión a formato YOLO
  renombra las imágenes con índices correlativos, por lo que el nombre del
  archivo no identifica la clase.
- **Medición de latencia sincronizada.** Las operaciones de CUDA son asíncronas;
  sin `torch.cuda.synchronize()` antes de detener el cronómetro se mide el tiempo
  de encolado y no el de cómputo.
- **Modelo cargado una sola vez.** Recargar el modelo por imagen domina el tiempo
  de proceso y falsearía las métricas del lote.

---

## 📁 Estructura del proyecto

El repositorio separa la lógica reutilizable (`src/`), los puntos de entrada
(`scripts/`), los datos, los modelos y los artefactos generados:

```
proyecto_inspeccion_visual/
│
├── README.md                            # Este documento
├── requirements.txt                     # Dependencias
├── .gitignore                           # Exclusiones de control de versiones
│
├── src/                                 # Lógica reutilizable
│   ├── __init__.py
│   ├── config.py                        # Rutas y configuración global
│   └── inspection.py                     # Decisión (compartida con la app)
│
├── scripts/                             # Puntos de entrada ejecutables
│   ├── inspeccion_visual_yolov8.py      # Pipeline completo (6 fases)
│   ├── app.py                           # Aplicación web (Streamlit)
│   ├── generar_dataset_sintetico.py     # Generación del conjunto de datos
│   ├── generar_figuras_informe.py       # Figuras del informe
│   ├── generar_informe_word.py          # Informe en Word (APA 7)
│   └── exportar_informe_txt.py          # Informe en texto plano
│
├── docs/                                # Documentos entregables
│   ├── proyecto_inspeccion_visual.docx  # Informe académico (Word)
│   └── proyecto_inspeccion_visual.txt   # Informe académico (texto)
│
├── data/                                # Datos (regenerables, no versionados)
│   ├── mvtec_ad/bottle/                 # Réplica sintética del conjunto
│   │   ├── train/{good,<defectos>}/     # Incluye defectos etiquetados
│   │   ├── test/{good,<defectos>}/
│   │   └── ground_truth/<defecto>/      # Máscaras de segmentación
│   └── yolo_mvtec/                      # Conjunto convertido a formato YOLO
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       └── data.yaml
│
├── models/
│   ├── pretrained/yolov8s.pt            # Pesos iniciales (COCO)
│   ├── trained/best_yolov8_inspection.pt# Mejor modelo entrenado
│   └── exports/                         # Modelos ONNX y TorchScript
│
├── results/                             # Artefactos de medición
│   ├── dataset_summary.json             # Resumen del conjunto
│   ├── split_composition.json           # Instancias por clase y partición
│   ├── training_metrics.json            # Métricas de entrenamiento
│   ├── evaluation_metrics.json          # Métricas globales y por clase
│   ├── confusion_matrix.json            # Matriz de confusión
│   ├── latency_metrics.json             # Estadísticos y muestras de latencia
│   ├── batch_inspection_report.json     # Inspección por lotes
│   ├── figuras/                         # 12 figuras del informe
│   ├── inspections/                     # Imágenes inspeccionadas
│   └── validation/                      # Curvas y predicciones de la validación
│
└── runs/detect/yolov8_inspection/       # Registro de Ultralytics
    ├── results.csv                      # Métricas por época
    ├── args.yaml                        # Hiperparámetros efectivos
    ├── weights/{best,last}.pt
    └── *.png                            # Curvas de entrenamiento
```

### Módulos compartidos

| Módulo | Responsabilidad |
|--------|-----------------|
| `src/config.py` | Rutas absolutas ancladas a la raíz y configuración del pipeline |
| `src/inspection.py` | Decisión de aprobación/rechazo, usada por el pipeline y la app |

> ℹ️ La regla de decisión vive en un único lugar. Antes existían dos
> implementaciones paralelas —una en el pipeline y otra en la aplicación web—
> que podían divergir al ajustar umbrales.

---

## 📦 Requisitos

- **Python** 3.10 o superior (probado en 3.14)
- **pip** actualizado
- *(Opcional)* **GPU NVIDIA con CUDA** para acelerar el entrenamiento

Dependencias principales: `ultralytics`, `torch`, `torchvision`,
`opencv-python`, `streamlit`, `numpy`, `pandas`, `matplotlib`, `seaborn`,
`scikit-learn`, `Pillow`, `python-docx`, `tqdm`.

---

## 🚀 Instalación

### 1. Crear y activar un entorno virtual (recomendado)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. GPU (opcional, muy recomendable)

`requirements.txt` instala la variante de PyTorch disponible por defecto, que en
Windows suele ser **solo CPU**. Para entrenar con GPU, instala un par
`torch`/`torchvision` compilado con CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verifica que la GPU se detecta correctamente:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> ⚠️ **Sobre las versiones.** `torch` y `torchvision` deben ser un par
> compatible. Si `torchvision` resuelve a una versión muy antigua, fija ambas
> explícitamente, por ejemplo:
> `pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128`

El pipeline detecta el dispositivo automáticamente (`cuda` si está disponible,
`cpu` en caso contrario), por lo que el mismo código funciona en ambos entornos.

---

## 📥 Obtención del conjunto de datos

El pipeline espera los datos en `data/mvtec_ad/bottle/`. Hay dos caminos:

**Opción A — Réplica sintética (recomendada para empezar):**

```bash
python scripts/generar_dataset_sintetico.py
```

Genera un conjunto reproducible (semilla fija) con imágenes normales y los tres
tipos de defecto, cada uno con su máscara binaria, respetando la estructura de
directorios del MVTec AD. Permite ejecutar y evaluar el sistema completo sin
depender de la descarga del corpus original.

| Partición | Normal | Grieta grande | Grieta pequeña | Contaminación | Total |
|-----------|-------:|--------------:|---------------:|--------------:|------:|
| Entrenamiento | 110 | 32 | 32 | 32 | 206 |
| Prueba | 30 | 16 | 16 | 16 | 78 |

La partición de prueba se divide además en validación (25 %) y prueba (75 %) de
forma **estratificada por clase**.

**Opción B — Conjunto MVTec AD oficial:**

1. Solicítalo en el sitio oficial de MVTec:  
   <https://www.mvtec.com/company/research/datasets/mvtec-ad>
2. Extrae la categoría `bottle` en `data/mvtec_ad/` con esta estructura:

```
data/mvtec_ad/
└── bottle/
    ├── train/good/          # Imágenes normales
    ├── test/
    │   ├── good/
    │   ├── broken_large/
    │   ├── broken_small/
    │   └── contamination/
    └── ground_truth/        # Máscaras de segmentación
```

> ⚠️ El MVTec AD oficial solo aporta imágenes normales en `train/`. Para entrenar
> un detector supervisado hay que **añadir ejemplos defectuosos etiquetados** a
> esa partición (véase la [nota metodológica](#-descripción-del-proyecto)).

---

## ▶️ Uso

### Pipeline completo

Ejecuta las seis fases (preparación → entrenamiento → evaluación → inspección →
exportación):

```bash
python scripts/inspeccion_visual_yolov8.py
```

### Recalcular sin reentrenar

Recalcula métricas, matriz de confusión, inspección por lotes y exportación
reutilizando el modelo ya entrenado:

```bash
python scripts/inspeccion_visual_yolov8.py evaluar
```

Útil tras modificar la lógica de evaluación o para regenerar artefactos sin
esperar de nuevo al entrenamiento.

### Generar el informe

```bash
python scripts/generar_figuras_informe.py   # 12 figuras a partir de results/*.json
python scripts/generar_informe_word.py      # Documento Word (APA 7) en docs/
python scripts/exportar_informe_txt.py      # Versión en texto plano en docs/
```

> ℹ️ Las cifras del informe se leen de los archivos JSON de `results/`, de modo
> que el documento y la evidencia experimental no pueden divergir.

### Parámetros configurables

Se ajustan en `src/config.py`, que además centraliza todas las rutas del
proyecto:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `category` | `bottle` | Categoría del conjunto MVTec AD |
| `img_size` | `640` | Tamaño de entrada de YOLOv8 |
| `epochs` | `100` | Máximo de épocas (con parada temprana) |
| `batch_size` | `16` | Tamaño de lote |
| `val_ratio` | `0.25` | Proporción de validación dentro de la partición de prueba |
| `seed` | `42` | Semilla del particionado estratificado |
| `model_size` | `yolov8s.pt` | Variante del modelo (n/s/m/l/x) |
| `confidence_threshold` | `0.25` | Umbral de confianza para la decisión |
| `iou_threshold` | `0.45` | Umbral de IoU para la supresión de no máximos |

El pipeline es **idempotente**: limpia las particiones derivadas antes de
repartirlas, de modo que ejecuciones sucesivas no acumulan archivos ni desplazan
las proporciones.

---

## 🌐 Aplicación web

```bash
streamlit run scripts/app.py
```

Disponible en **<http://localhost:8501>**.

| Pestaña | Descripción |
|---------|-------------|
| 📷 **Inspección individual** | Sube una imagen y obtén la decisión con el defecto localizado |
| 📁 **Inspección por lotes** | Analiza varias imágenes y descarga el reporte en JSON |
| 📈 **Dashboard de métricas** | Métricas de evaluación, latencia y entrenamiento |
| 📚 **Documentación** | Guía integrada del sistema y de los tipos de defecto |

### Flujo de uso

1. En la barra lateral, pulsa **🚀 Cargar Modelo**.
2. Ajusta los umbrales de confianza e IoU si lo necesitas.
3. En **Inspección individual**, sube una imagen y pulsa **🔍 Iniciar Inspección**.
4. Pulsa **📊 Cargar Métricas** para ver el panel de resultados.

> Si el modelo entrenado no existe, la aplicación muestra un aviso y usa
> `models/pretrained/yolov8s.pt` como respaldo. En ese caso las detecciones no son
> fiables, porque el modelo base no conoce las clases de defecto del proyecto.

---

## 📊 Resultados

Medición sobre la partición de prueba independiente (58 imágenes, 36 instancias
anotadas), con umbral de confianza 0.25 e IoU 0.45. Hardware: NVIDIA RTX 4060.

### Métricas globales

| Métrica | Valor | Lectura |
|---------|------:|---------|
| **mAP@0.5** | **0.926** | Localización y clasificación correctas con IoU ≥ 0.5 |
| **mAP@0.5:0.95** | 0.654 | Consistencia en umbrales de solapamiento estrictos |
| **Precisión** | 0.835 | Proporción de detecciones correctas |
| **Exhaustividad** | 0.972 | Prácticamente todos los defectos reales detectados |
| **F1-score** | 0.884 | Equilibrio entre precisión y exhaustividad |

### Desempeño por clase

| Clase | Precisión | Exhaustividad | F1 | AP@0.5 |
|-------|----------:|--------------:|---:|-------:|
| Grieta grande | 0.582 | **1.000** | 0.736 | 0.873 |
| Grieta pequeña | **1.000** | 0.917 | 0.957 | 0.915 |
| Contaminación | 0.923 | **1.000** | 0.960 | 0.989 |

### Matriz de confusión

Filas = instancia real, columnas = predicción.

| Real ↓ / Predicho → | Grieta grande | Grieta pequeña | Contaminación | Fondo |
|---------------------|--------------:|---------------:|--------------:|------:|
| Grieta grande | 12 | 0 | 0 | 0 |
| Grieta pequeña | 1 | 10 | 0 | 1 |
| Contaminación | 0 | 0 | 12 | 0 |
| Fondo (sin real) | 8 | 1 | 1 | — |

- **Ninguna pieza normal genera detección** y **ninguna instancia de grieta grande
  o contaminación queda sin detectar**.
- Las 8 detecciones de la fila *Fondo* no provienen de piezas conformes, sino de
  grietas extensas que reciben más de una caja al fragmentarse en tramos
  contiguos. Es un error de **delimitación**, no de detección.

> **Lectura operativa.** El sistema no aprueba piezas defectuosas ni rechaza
> piezas conformes. Su debilidad se concentra en la precisión de la caja sobre
> fracturas ramificadas, un aspecto que penaliza la métrica pero no compromete la
> función de control de calidad.

### Latencia de inferencia

100 inferencias consecutivas, con calentamiento previo y sincronización de GPU.

| Estadístico | Valor |
|-------------|------:|
| Media | **12.54 ms** |
| Mediana | 12.23 ms |
| Desviación estándar | 2.32 ms |
| Percentil 95 | 16.83 ms |
| Percentil 99 | **20.66 ms** |
| Máxima | 22.92 ms |
| Rendimiento | **79.8 FPS** |

El requisito industrial adoptado es una latencia inferior a 30 ms por imagen. Se
cumple tanto en la media como en el **percentil 99**, es decir, incluso en el caso
adverso extremo.

### Inspección por lotes

Lote de 30 imágenes con muestreo estratificado por clase.

| Tipo de pieza | Piezas | Aprobadas | Rechazadas |
|---------------|-------:|----------:|-----------:|
| Normal | 12 | **12** | 0 |
| Grieta grande | 6 | 0 | 6 |
| Grieta pequeña | 6 | 1 | 5 |
| Contaminación | 6 | 0 | 6 |
| **Total** | **30** | **13** | **17** |

Todas las piezas conformes se aprobaron y la práctica totalidad de las
defectuosas se rechazó. El único caso favorable correspondió a una grieta
pequeña, coherente con ser la clase de detección más difícil.

### Coste de entrenamiento

| Aspecto | Valor |
|---------|------:|
| Épocas ejecutadas | 36 (parada temprana, máximo 100) |
| Mejor mAP@0.5 en validación | 0.995 (época 21) |
| Tiempo total | 16.1 min (GPU RTX 4060) |

### Comparativa: efecto del particionado

| Configuración | mAP@0.5 | Precisión | Exhaustividad |
|---------------|--------:|----------:|--------------:|
| Sin defectos en entrenamiento | 0.313 | 0.417 | 0.313 |
| **Con defectos en entrenamiento** | **0.926** | **0.835** | **0.972** |

La primera configuración reproduce la herencia del diseño no supervisado del
MVTec AD: el modelo nunca observa un defecto y, por tanto, no puede aprender a
localizarlo.

### Artefactos generados

| Archivo | Contenido |
|---------|-----------|
| `results/dataset_summary.json` | Inventario del conjunto por partición |
| `results/split_composition.json` | Instancias anotadas por clase y partición |
| `results/training_metrics.json` | Duración, épocas y métricas finales |
| `results/evaluation_metrics.json` | Métricas globales y desagregadas por clase |
| `results/confusion_matrix.json` | Matriz de confusión sobre prueba |
| `results/latency_metrics.json` | Estadísticos y mediciones individuales |
| `results/batch_inspection_report.json` | Decisiones del lote con desglose por tipo |
| `results/figuras/` | 12 figuras del informe |
| `models/trained/best_yolov8_inspection.pt` | Modelo entrenado |
| `models/exports/best_yolov8_inspection.onnx` | Exportación ONNX |
| `models/exports/best_yolov8_inspection.torchscript` | Exportación TorchScript |

---

## 🔬 Reproducibilidad

Todas las cifras anteriores se obtienen ejecutando, en este orden:

```bash
python scripts/generar_dataset_sintetico.py     # Datos (semilla fija)
python scripts/inspeccion_visual_yolov8.py      # Entrenamiento + evaluación + exportación
python scripts/generar_figuras_informe.py       # Figuras
python scripts/generar_informe_word.py          # Informe
python scripts/exportar_informe_txt.py          # Versión en texto
```

El particionado y el aumento de datos usan semillas fijas, y las versiones de las
librerías están acotadas en `requirements.txt`. Las cifras de latencia dependen
del hardware y del estado de la máquina: para reproducirlas conviene **cerrar
otros procesos que usen la GPU** (por ejemplo, un servidor de Streamlit abierto
en paralelo puede multiplicar la latencia medida).

---

## 🛠️ Solución de problemas

| Problema | Causa y solución |
|----------|------------------|
| `ModuleNotFoundError: No module named 'ultralytics'` | El entorno virtual no está activo o faltan dependencias. Ejecuta `pip install -r requirements.txt`. |
| `torch.cuda.is_available()` devuelve `False` | Se instaló la variante CPU. Reinstala con el índice CUDA (véase [Instalación](#-instalación)). |
| `torchvision` resuelve a una versión incompatible | Fija el par compatible: `pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128`. |
| Conjunto de datos no encontrado | Ejecuta `python scripts/generar_dataset_sintetico.py` o coloca el MVTec AD en `data/mvtec_ad/bottle`. |
| El modelo no detecta ningún defecto | Comprueba que la partición de entrenamiento contiene ejemplos defectuosos, no solo imágenes normales. |
| mAP muy bajo (≈ 0.3) pese a entrenar | Misma causa: el modelo nunca vio un defecto. Revisa la composición de `train/`. |
| Latencias irreales o irreproducibles | Las mediciones en GPU requieren sincronizar CUDA antes de detener el cronómetro. El pipeline ya lo hace; si mides por tu cuenta, añade `torch.cuda.synchronize()`. |
| Latencia muy superior a la documentada | Hay otros procesos usando la GPU. Cierra servidores o aplicaciones y vuelve a medir. |
| Resultados duplicados al reejecutar | El pipeline limpia las particiones derivadas antes de repartirlas. Si persiste, borra `data/yolo_mvtec/`. |
| `FileNotFoundError` al buscar el modelo entrenado | Verifica que el entrenamiento terminó y que `models/trained/best_yolov8_inspection.pt` existe. |
| La aplicación no abre | Comprueba que Streamlit está instalado y ejecuta desde la raíz: `streamlit run scripts/app.py`. |
| Puerto 8501 ocupado | Usa otro puerto: `streamlit run scripts/app.py --server.port 8502`. |
| `ModuleNotFoundError: No module named 'src'` | Ejecuta los scripts desde la raíz del proyecto, o usa la ruta completa (`python scripts/<script>.py`). |

---

## 📖 Referencias

- Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). The
  MVTec anomaly detection dataset: A comprehensive real-world dataset for
  unsupervised anomaly detection. *International Journal of Computer Vision,
  129*(4), 1038–1059. <https://doi.org/10.1007/s11263-020-01400-4>
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Versión 8.0)
  [Software de cómputo]. GitHub. <https://github.com/ultralytics/ultralytics>
- Redmon, J., & Farhadi, A. (2018). *YOLOv3: An incremental improvement*. arXiv.
  <https://doi.org/10.48550/arXiv.1804.02767>
- Yuan, Y., Xia, Y., & Xu, G. (2025). Metal surface defect detection using
  SLF-YOLO enhanced YOLOv8 model. *Scientific Reports, 15*, Artículo 94936.
  <https://doi.org/10.1038/s41598-025-94936-9>

---

## 📄 Informe académico

La memoria del proyecto está en **`docs/proyecto_inspeccion_visual.docx`**, con
versión en texto plano en **`docs/proyecto_inspeccion_visual.txt`**. Contiene:

- Portada, resumen y palabras clave
- Índice y listas de tablas y figuras
- Introducción con objetivos general y específicos
- Descripción del problema y del conjunto de datos
- Metodología: arquitectura, hiperparámetros y métricas
- Resultados y discusión (7 tablas y 12 figuras)
- Análisis del efecto del particionado del conjunto de datos
- Conclusiones, limitaciones y trabajo futuro
- Referencias en APA 7.ª edición
- Apéndices de estructura, reproducibilidad y artefactos

> ℹ️ El informe se **genera automáticamente** desde los archivos JSON de
> `results/`. Cualquier reejecución del pipeline produce un documento coherente
> con la evidencia experimental.

---

## 📜 Licencia y uso

El código de este repositorio se distribuye con fines **académicos y educativos**.
El conjunto **MVTec AD** es propiedad de MVTec Software GmbH y se distribuye bajo
licencia *Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)*, por lo que **no admite uso comercial**. La réplica sintética
incluida se genera localmente y no contiene datos del conjunto original.

---

**Materia:** Aplicaciones Prácticas de IA  
**Tecnologías:** Python · YOLOv8 · PyTorch · OpenCV · Streamlit  
**Conjunto de datos:** réplica sintética de MVTec AD (categoría *bottle*)  
**Modelo:** YOLOv8s
