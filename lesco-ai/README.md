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
│   ├── feature_extraction.py
│   ├── hand_tracker.py
│   ├── model_utils.py
│   ├── predict_live.py
│   ├── runtime_config.py
│   ├── record_sign.py
│   ├── train_model.py
│   └── config.py
├── dataset/
├── raw_data/
├── processed_data/
├── models/
├── docs/
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

Modo debug:

```bash
python src/predict_live.py --debug
```

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
python src/predict_live.py --debug --save-clip clips/demo.npy
```

## Pruebas

```bash
python -m unittest discover -s tests
```
