# LESCO-AI

Proyecto Python para reconocer señas LESCO usando visión artificial, landmarks de
MediaPipe y un modelo temporal entrenado con features relativos a la mano.

## 1) Crear entorno virtual

Desde la carpeta `lesco-ai`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Instalar dependencias

```bash
pip install -r requirements.txt
```

## 3) Ejecutar prueba de cámara y manos

```bash
cd src
python3 camera_test.py
```

Se abrirá una ventana con la cámara en vivo y se dibujarán los landmarks de la mano cuando se detecten.

Para salir, presiona la tecla `q`.

## 4) Si la cámara no abre

- Verifica que no esté siendo usada por otra aplicación.
- Revisa el índice de cámara en `src/config.py` (`CAMERA_INDEX = 0`, prueba con `1` o `2` si es necesario).
- Asegúrate de tener permisos para acceder al dispositivo de video en Linux.
- Si usas webcam USB, prueba desconectar y reconectar.

## Pipeline actual

El reconocimiento no usa directamente las coordenadas absolutas de MediaPipe.
Cada secuencia se convierte con un extractor compartido en:

- coordenadas relativas a la muñeca;
- escala normalizada por tamaño de palma;
- distancias de huesos de la mano;
- velocidad y aceleración temporal de los landmarks normalizados;
- trayectoria de la muñeca relativa al inicio de la seña.

El entrenamiento y la predicción usan el mismo módulo:
`src/feature_extraction.py`.

## Arquitectura actual

El código de reconocimiento en vivo está dividido por responsabilidades para
evitar que el punto de entrada mezcle cámara, segmentación, predicción, debug y
salida hacia Godot.

- `src/predict_live.py`: punto de entrada. Lee argumentos, carga configuración,
  prepara rutas de `godot_bridge`, ejecuta clips offline y coordina el loop de
  cámara en vivo.
- `src/live_session.py`: estado mutable de la sesión en vivo. Contiene
  `LandmarkClipRecorder`, la máquina de estados `WAITING`, `MOVING` y
  `POSSIBLE_PAUSE`, el cierre por ausencia de manos, publicación de frames,
  recarga con `c` y salida con `q`.
- `src/segment_prediction.py`: clasificación de segmentos ya cortados,
  predicciones top 3 crudas, aceptación por confianza y acumulación de segmentos
  aceptados para construir la oración final.
- `src/continuous_recognition.py`: reconocimiento continuo sobre clips completos.
  Mantiene `ContinuousRecognizer`, `PrototypeLibrary`, `SentenceBuilder`,
  predicción por ventanas, scoring visual/prototipos y construcción de oración.
- `src/detection_grouping.py`: tipos y reglas temporales para ventanas y
  detecciones, incluyendo IoU, overlap, comparación de eventos y unión de
  ventanas repetidas.
- `src/hand_tracker.py`: integración con MediaPipe, extracción de landmarks,
  dibujo de manos y asignación estable de slots para una o dos manos.
- `src/debug_view.py`: overlay de OpenCV y escritura de
  `godot_bridge/debug_response.txt`.
- `src/sign_video_bridge.py`: comunicación por archivos con Godot, incluyendo
  `output.txt` para oraciones detectadas y reproducción de videos de señas hacia
  `sign_video_frame.jpg`.
- `src/runtime_config.py` y `src/config_ui.py`: carga, validación, overrides por
  argumentos y edición interactiva de configuración.
- `src/model_utils.py`, `src/dataset_utils.py` y `src/feature_extraction.py`:
  carga de modelos/labels, lectura de dataset y pipeline de features compartido.

## Estructura del proyecto

```text
lesco-ai/
├── src/
│   ├── __init__.py
│   ├── camera_test.py
│   ├── config_ui.py
│   ├── continuous_recognition.py
│   ├── dataset_utils.py
│   ├── debug_view.py
│   ├── detection_grouping.py
│   ├── feature_extraction.py
│   ├── hand_tracker.py
│   ├── live_session.py
│   ├── model_utils.py
│   ├── predict_live.py
│   ├── runtime_config.py
│   ├── record_sign.py
│   ├── record_sign_video.py
│   ├── segment_prediction.py
│   ├── sign_video_bridge.py
│   ├── train_model.py
│   └── config.py
├── dataset/
├── dataset_legacy_one_hand/
├── godot_bridge/
├── models/
├── videos_database/
├── requirements.txt
└── README.md
```

## Entrenar y reconocer

```bash
python src/train_model.py
python src/predict_live.py
python src/record_sign.py --label gracias
```

## Reconocimiento continuo

Modo normal de presentación:

```bash
python src/predict_live.py
```

El sistema espera a que aparezcan manos de forma estable, graba mientras se
realizan varias señas, finaliza cuando las manos desaparecen durante el tiempo
configurado, procesa la secuencia y vuelve automáticamente a esperar.

El flujo en vivo es:

1. `predict_live.py` abre la cámara y crea `LiveRecognitionSession`.
2. `HandTracker` detecta landmarks y `select_two_hand_slots` conserva slots
   estables para dos manos.
3. `LandmarkClipRecorder` corta segmentos según movimiento, pausa confirmada,
   duración mínima/máxima y ausencia de manos.
4. `SegmentPredictionBuffer` clasifica cada segmento, conserva el top 3 crudo
   para debug y acepta los segmentos que superan `min_confidence`.
5. Al cerrar la oración por ausencia de manos, los segmentos aceptados se pasan
   a `SentenceBuilder` para producir la oración final.
6. `sign_video_bridge.write_godot_output` escribe `godot_bridge/output.txt` y
   `debug_view.write_debug_response` escribe `godot_bridge/debug_response.txt`.

Nota sobre movimiento de salida: el recorder todavía marca internamente
`movement_exit` para describir que un clip se cerró durante la salida de manos,
pero esa marca ya no cancela la predicción. Si el segmento supera
`min_confidence`, se acepta y puede formar parte de la oración final.

Configuración local:

```bash
python src/predict_live.py --config
```

Procesar un clip de landmarks para pruebas:

```bash
python src/predict_live.py --input-npy clip.npy
```

Guardar clips de debug:

```bash
python src/predict_live.py --save-clip clips/demo.npy
```

Las predicciones crudas recientes del modo vivo se escriben en
`godot_bridge/debug_response.txt`.

La salida para Godot se mantiene por archivos:

- `godot_bridge/output.txt`: oración final, score visual y detecciones.
- `godot_bridge/debug_response.txt`: segmentos recientes, top 3 crudo y decisión
  de aceptación/rechazo por umbral.
- `godot_bridge/frame.jpg`: último frame del reconocimiento en vivo.
- `godot_bridge/sign_video_input.txt`: seña solicitada por Godot para reproducir
  video.
- `godot_bridge/sign_video_frame.jpg`: frame exportado del video solicitado.

## Pruebas

```bash
python -m unittest discover -s tests
```

En este entorno local también se puede usar:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Si `pytest` no está instalado, usa `unittest`; las pruebas actuales están
escritas con `unittest`.
