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
python src/camera_test.py
```

Se abrirá una ventana con la cámara en vivo y se dibujarán los landmarks de la mano cuando se detecten.

Para salir, presiona la tecla `q`.

## 4) Si la cámara no abre

- Verifica que no esté siendo usada por otra aplicación.
- Revisa el índice de cámara en `src/config/constants.py` (`CAMERA_INDEX = 0`, prueba con `1` o `2` si es necesario).
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
`src/vision/features.py`.

## Estructura de `src/`

La implementación está organizada por responsabilidad. Los flujos de datos se
ejecutan directamente como módulos de `src.data`; los lanzadores que permanecen
en `src/` corresponden a componentes interactivos de la aplicación.

```text
src/
├── config/          # constantes de cámara, configuración runtime y editor
├── vision/          # MediaPipe, slots de manos y extracción de features
├── recognition/     # modelo, segmentación, agrupación, sesión y predicción
├── data/            # dataset, grabación de muestras/videos y entrenamiento
├── integration/     # bridge de archivos y videos para Godot
├── diagnostics/     # prueba de cámara, overlay y salida de debug
├── predict_live.py  # lanzador del reconocimiento interactivo
├── camera_test.py
└── sign_video_bridge.py
```

Los módulos principales son:

- `src/recognition/predict_live.py`: entrada y coordinación de cámara o clips.
- `src/recognition/session.py`: máquina de estados y sesión en vivo.
- `src/recognition/segments.py`: clasificación y acumulación de segmentos.
- `src/recognition/continuous.py` y `grouping.py`: ventanas, detecciones,
  prototipos y construcción temporal de oraciones.
- `src/vision/features.py`: pipeline compartido de landmarks a features.
- `src/vision/hand_tracker.py`: MediaPipe y slots estables de manos.
- `src/config/runtime.py`: configuración editable del reconocimiento en vivo.
- `src/integration/sign_video_bridge.py`: archivos compartidos con Godot.

## Entrenar y reconocer

### Arrancar Prisma completo

La web, el reconocimiento en vivo y el puente de videos se pueden iniciar y
cerrar juntos desde una sola terminal:

```bash
python prisma.py
```

`Ctrl+C` solicita primero un cierre ordenado de los tres procesos. Los comandos
individuales indicados abajo se conservan para depuración y mantenimiento.

```bash
python -m src.data.train_model
python src/predict_live.py
python -m src.data.record_sign --label gracias
python -m src.data.record_sign_video --label gracias
```

Para una seña que necesariamente use ambas manos, valida la muestra de forma
explícita sin asociar esa regla a una etiqueta concreta:

```bash
python -m src.data.record_sign --label nombre_de_la_seña --require-two-hands
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

## Configuración del reconocimiento

### Configuración runtime centralizada

Estos valores están centralizados en `LiveRecognitionConfig`, aproximadamente en
`src/config/runtime.py:15-25`. Se pueden editar con
`python src/predict_live.py --config`; si existe,
`config/live_runtime_config.json` reemplaza los defaults. Algunos también admiten
override por argumentos (`--min-confidence`, `--stride`, `--record-seconds`,
`--no-prototypes` y `--save-clip`).

| Parámetro | Valor actual | Qué controla | Efecto al subirlo o bajarlo |
|---|---:|---|---|
| `min_confidence` | `0.60` | Confianza mínima para aceptar un segmento; también se pasa al reconocimiento de clips. | Subir: menos falsos positivos y más rechazos. Bajar: más detecciones y más riesgo de error. |
| `stride` | `4` frames | Paso entre ventanas del reconocimiento continuo/offline. El flujo en vivo segmentado clasifica el segmento completo y no usa este paso. | Subir: menos ventanas y menor costo, con menor resolución temporal. Bajar: más ventanas y mayor costo. |
| `no_hands_timeout_seconds` | `0.9 s` | Ausencia de manos necesaria para cerrar una oración. | Subir: tolera ausencias largas. Bajar: cierra antes y puede cortar oraciones. |
| `movement_threshold` | `0.14` | Movimiento normalizado mínimo para iniciar/continuar un segmento. | Subir: exige gestos más marcados. Bajar: detecta movimientos sutiles, incluido ruido. |
| `pause_frames` | `3` | Frames quietos para confirmar una pausa entre segmentos. | Subir: une pausas cortas. Bajar: separa señas antes y puede fragmentarlas. |
| `min_clip_seconds` | `0.1 s` | Duración mínima aceptada para un clip. | Subir: descarta movimientos breves. Bajar: admite clips más cortos y ruido. |
| `max_clip_seconds` | `12.0 s` | Duración máxima antes de forzar el cierre del segmento. | Subir: permite gestos largos. Bajar: reduce latencia/memoria, pero puede cortar gestos. |
| `use_prototypes` | `true` | Carga prototipos visuales calculados desde el dataset. | Desactivar reduce arranque y comparación visual auxiliar. |
| `show_landmarks` | `true` | Dibuja landmarks sobre el frame publicado. | Solo afecta visualización y algo de costo de dibujo. |
| `save_debug_clips` | `false` | Guarda clips capturados para diagnóstico. | Activarlo aumenta escrituras y uso de disco. |
| `save_clip_dir` | `clips/debug` | Directorio de clips de diagnóstico. | No altera la predicción. |

### Cámara y MediaPipe

Centralizados en `src/config/constants.py`. El constructor `HandTracker` usa
estas mismas constantes como defaults y sigue permitiendo overrides explícitos.

| Parámetro | Valor actual | Qué controla | Efecto al subirlo o bajarlo |
|---|---:|---|---|
| `CAMERA_INDEX` | `0` | Dispositivo de cámara. | Cambiarlo selecciona otra webcam. |
| `FRAME_WIDTH` | `640` | Ancho solicitado de captura. | Subir puede mejorar detalle y aumentar costo. |
| `FRAME_HEIGHT` | `480` | Alto solicitado de captura. | Subir puede mejorar detalle y aumentar costo. |
| `MAX_NUM_HANDS` | `2` | Máximo de manos para MediaPipe. | Bajarlo a `1` impide capturar correctamente señas de dos manos. |
| `MIN_DETECTION_CONFIDENCE` | `0.6` | Umbral de detección inicial de MediaPipe. | Subir reduce detecciones dudosas; bajar detecta más manos con más falsos positivos. |
| `MIN_TRACKING_CONFIDENCE` | `0.5` | Umbral de seguimiento de MediaPipe. | Subir exige tracking más estable; bajar tolera frames más inciertos. |

### Features, frames y segmentación interna

Estos valores están dispersos y no forman parte del JSON runtime.

| Parámetro | Valor actual | Archivo aproximado | Efecto |
|---|---:|---|---|
| `SEQUENCE_LENGTH` | `30` frames | `src/config/constants.py` | Longitud temporal de entrada del LSTM. Cambiarlo exige volver a entrenar el modelo. |
| `HANDS_PER_FRAME` | `2` | `src/config/constants.py` | Cantidad de slots de manos del modelo. Cambiarlo altera el shape y exige reentrenamiento. |
| `LANDMARKS_PER_HAND` | `21` | `src/config/constants.py` | Landmarks MediaPipe por mano; forma parte del contrato del modelo. |
| `STATIC_SIGNATURE_DOMINANCE_THRESHOLD` | `20.0%` | `src/vision/features.py:25` | Aceptación de una firma estática. Subir exige mayor separación; bajar acepta firmas más ambiguas. |
| `DEFAULT_FPS` | `30.0` | `src/config/runtime.py` | FPS de respaldo compartido por cámara, grabación de video y bridge. |
| `USE_ACCELERATION_ACTIVITY` | `true` | `src/recognition/session.py:26` | Habilita el cálculo de aceleración en actividad. Con la fórmula actual no supera al score de movimiento. |
| `USE_STATIC_LANDMARK_SIGNATURES` | `false` | `src/recognition/session.py:27` | Habilita firmas estáticas al confirmar pausas. Actualmente está desactivado. |
| `ACCELERATION_ACTIVITY_WEIGHT` | `0.5` | `src/recognition/session.py:28` | Peso previsto para aceleración del segmentador; actualmente no cambia el máximo final. |

### Agrupación, tolerancias y scoring

Son constantes internas, principalmente en `src/recognition/grouping.py:12-17`
y `src/recognition/continuous.py:180-365`. No están centralizadas todavía.

| Parámetro | Valor actual | Qué controla | Efecto al subirlo o bajarlo |
|---|---:|---|---|
| `SAME_WORD_IOU_THRESHOLD` | `0.30` | IoU para unir ventanas de la misma clase. | Subir exige mayor coincidencia; bajar fusiona más. |
| `SAME_WORD_OVERLAP_RATIO_THRESHOLD` | `0.50` | Solapamiento respecto de la ventana corta. | Subir separa más eventos; bajar une más. |
| `SAME_WORD_CENTER_DISTANCE_FACTOR` | `0.75` | Distancia máxima entre centros relativa a la duración. | Subir permite agrupar eventos más alejados. |
| `SAME_WORD_MAX_CHAIN_SPAN_FACTOR` | `1.90` | Longitud máxima de una cadena de ventanas repetidas. | Subir favorece cadenas más largas; bajar las divide antes. |
| `SAME_WORD_CHAIN_START_FACTOR` | `0.85` | Separación necesaria para considerar una segunda ejecución. | Subir hace menos probable dividir la cadena. |
| `max_gap_frames` | `8` | Hueco permitido al comparar/agrupar eventos de la misma clase. | Subir fusiona eventos más separados. |
| `cross_label_iou` | `0.45` | IoU para descartar una clase competidora sobre los mismos frames. | Subir conserva más candidatos; bajar suprime más. |
| confianza fuerte sin soporte | `0.80` | Permite conservar una detección respaldada por una sola ventana. | Subir exige más confianza; bajar admite más candidatos aislados. |
| `beam_width` | `5` | Cantidad de hipótesis del constructor temporal. | Subir explora más alternativas y consume más CPU. |
| `visual_weight` | `4.0` | Peso de evidencia visual en el beam. | Subir hace dominar más la evidencia visual frente a penalizaciones temporales. |
| `skip_penalty` | `0.25` | Penalización usada al ordenar hipótesis que omiten detecciones. | Subir desalienta omitir candidatos. |
| penalización de solapamiento | `-0.55 × IoU` | Penaliza detecciones consecutivas sobre los mismos frames. | Aumentar la magnitud separa más agresivamente. |
| penalización de repetición cercana | `-1.2` | Penaliza repetir la misma clase sin separación temporal suficiente. | Aumentar la magnitud elimina más repeticiones. |
| mezcla estática/modelo | `50% / 50%` | Confianza del segmento cuando existe firma estática válida. | Solo aplica si se habilitan firmas estáticas. |
| top de ventanas / debug | `2 / 3` | Candidatos por ventana continua y predicciones mostradas por segmento. | El top 3 de debug no modifica la decisión. |

Los defaults de reconocimiento continuo usan ahora `WINDOW_STRIDE=4` y
`MIN_CONFIDENCE=0.60` desde `src/config/runtime.py`, igual que
`LiveRecognitionConfig`. Ambos pueden seguir sobrescribiéndose explícitamente.

### Grabación y entrenamiento

| Parámetro | Valor actual | Archivo aproximado | Efecto |
|---|---:|---|---|
| `COUNTDOWN_SECONDS` / `FRAMES_PER_SAMPLE` | `3.0 s / 20` | `src/data/record_sign.py` | Espera antes de cada captura y cantidad exacta de frames válidos guardados por muestra. |
| `MAX_MISSING_TWO_HAND_RATIO` | `0.25` | `src/data/record_sign.py:24` | Proporción máxima de frames incompletos que se pueden rellenar. Subir acepta más datos interpolados. |
| `--require-two-hands` | desactivado | `src/data/record_sign.py:37-41` | Exige ambas manos sin asociar la regla a una seña concreta. |
| epochs / batch / test | `30 / 16 / 0.2` | `src/data/train_model.py:25-27` | Parámetros de entrenamiento; no cambian una sesión ya cargada. |
| muestras/clases mínimas | `10 / 2` | `src/data/train_model.py:18-19` | Validación mínima del dataset antes de entrenar. |

Configuraciones que convendría centralizar en una fase posterior:

- tolerancias de agrupación y pesos de scoring;
- flags y pesos experimentales de actividad/firmas estáticas;
- parámetros exclusivos de grabación y entrenamiento si llegan a necesitarse
  desde más de un módulo.

Esta limpieza no cambió los valores efectivos utilizados por el flujo principal.

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

## Prueba de interfaz web local

La API web reutiliza `godot_bridge/output.txt`, el mismo resultado que ya
publica el reconocimiento para Godot. Por eso no modifica el loop de cámara ni
el modelo y ambos clientes pueden probarse en paralelo.

En dos terminales, con el entorno virtual activo, ejecuta:

```bash
python src/predict_live.py
python web/web_api.py
```

Abre `http://127.0.0.1:5000/`. La página consulta cada 500 ms
`GET /resultado`, cuya respuesta tiene esta forma:

```json
{
  "seña": "HOLA",
  "confianza": 0.91
}
```

Mientras todavía no exista una predicción confiable, `seña` será una cadena
vacía y `confianza` será `null`. La API escucha solamente en `127.0.0.1:5000`.
