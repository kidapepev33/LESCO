# LESCO-AI

Base inicial de un proyecto modular en Python para traducir LESCO usando visión artificial.

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

## Estructura del proyecto

```text
lesco-ai/
├── src/
│   ├── __init__.py
│   ├── camera_test.py
│   ├── hand_tracker.py
│   └── config.py
├── dataset/
├── raw_data/
├── processed_data/
├── models/
├── docs/
├── requirements.txt
└── README.md
```

python3 -m venv .venv

source .venv/bin/activate
deactivated

python src/train_model.py

python src/predict_live.py

python src/record_sign.py --label gracias


## PLEASE WORK

python3 --version
pip3 --version

cd lesco-ai
source .venv/bin/activate
python3 src/predict_live.py