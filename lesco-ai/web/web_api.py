"""API web local para consultar el último resultado del reconocedor LESCO."""

from __future__ import annotations

import re
import time
from pathlib import Path

from flask import Flask, Response, jsonify, make_response, send_file, send_from_directory


WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
DEFAULT_RESULT_PATH = PROJECT_ROOT / "godot_bridge" / "output.txt"
DEFAULT_FRAME_PATH = PROJECT_ROOT / "godot_bridge" / "frame.jpg"

SENTENCE_PATTERN = re.compile(r"^Oración:\s*(.*)$", re.MULTILINE)
CONFIDENCE_PATTERN = re.compile(r"^Score visual:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
MJPEG_BOUNDARY = b"frame"


def stream_latest_frame(frame_path: Path, poll_interval: float = 0.01):
    """Emite el JPEG más reciente cada vez que cambia, sin mantener una cola."""
    last_mtime_ns: int | None = None

    try:
        while True:
            try:
                before = frame_path.stat()
                if before.st_mtime_ns == last_mtime_ns:
                    time.sleep(poll_interval)
                    continue

                jpeg = frame_path.read_bytes()
                after = frame_path.stat()
            except (FileNotFoundError, OSError):
                time.sleep(poll_interval)
                continue

            # cv2.imwrite reemplaza el contenido del archivo. Si coincidimos con
            # esa escritura, esperamos la siguiente vuelta en vez de enviar un
            # JPEG incompleto.
            stable_read = (
                before.st_mtime_ns == after.st_mtime_ns
                and before.st_size == after.st_size == len(jpeg)
            )
            if not stable_read or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                time.sleep(poll_interval)
                continue

            last_mtime_ns = after.st_mtime_ns
            yield (
                b"--" + MJPEG_BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
    except GeneratorExit:
        # Flask cierra el generador cuando el navegador abandona la página.
        return


def read_current_result(result_path: Path = DEFAULT_RESULT_PATH) -> dict[str, str | float | None]:
    """Convierte la salida compartida con Godot al contrato JSON de la web."""
    try:
        contents = result_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeError):
        return {"seña": "", "confianza": None}

    sentence_match = SENTENCE_PATTERN.search(contents)
    confidence_match = CONFIDENCE_PATTERN.search(contents)
    sentence = sentence_match.group(1).strip() if sentence_match else ""
    confidence = float(confidence_match.group(1)) if confidence_match else None
    return {"seña": sentence, "confianza": confidence}


def create_app(result_path: Path = DEFAULT_RESULT_PATH, frame_path: Path = DEFAULT_FRAME_PATH) -> Flask:
    """Crea la aplicación sin iniciar procesos de cámara ni de inferencia."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    page_files = {
        "lesco-a-texto": "lesco-a-texto.html",
        "texto-a-lesco": "texto-a-lesco.html",
        "nosotros": "nosotros.html",
        "ayuda": "ayuda.html",
    }

    @app.get("/<page_name>")
    def page(page_name: str):
        filename = page_files.get(page_name)
        if filename is None:
            return make_response("Página no encontrada", 404)
        return send_from_directory(WEB_DIR, filename)

    @app.get("/assets/<path:filename>")
    def web_assets(filename: str):
        """Sirve los recursos web desde el mismo origen que la API local."""
        response = make_response(send_from_directory(WEB_DIR / "assets", filename))
        # Durante el desarrollo evita que una versión anterior de app.js siga
        # reactivando el polling de /frame después de migrar al stream MJPEG.
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/resultado")
    def resultado():
        response = jsonify(read_current_result(result_path))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/frame")
    def frame():
        if not frame_path.is_file():
            return make_response("", 204)
        response = send_file(frame_path, mimetype="image/jpeg", conditional=False)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.get("/stream")
    def stream():
        response = Response(
            stream_latest_frame(frame_path),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
