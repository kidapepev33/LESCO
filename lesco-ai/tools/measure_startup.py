"""Temporary startup timing tool for predict_live.py.

Run from the project root:

    python tools/measure_startup.py
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from typing import Iterable

TIMEOUT_SECONDS = 10 * 60


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def result_path() -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"startup_results_{timestamp}.txt"


def pythonpath_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(root)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not current else os.pathsep.join([src_path, current])
    return env


def write_header(handle) -> None:
    handle.write("Computadora:\n")
    handle.write("Proyecto en disco interno/USB/red:\n")
    handle.write("Primera o segunda ejecucion:\n")
    handle.write("Uso alto de CPU/RAM/disco:\n")
    handle.write("Observaciones:\n")
    handle.write("\n")
    handle.flush()


def write_line(handle, line: str = "") -> None:
    handle.write(line + "\n")
    handle.flush()


def run_subprocess(
    name: str,
    command: list[str],
    handle,
    root: Path,
    results: list[dict[str, object]],
) -> None:
    started_at = dt.datetime.now()
    write_line(handle, f"== {name} ==")
    write_line(handle, f"Inicio: {started_at.isoformat(timespec='seconds')}")
    write_line(handle, "Comando: " + " ".join(command))
    print(f"[startup] {name}...")

    start = time.perf_counter()
    timeout = False
    ok = False
    stdout = ""
    stderr = ""
    error = ""
    returncode: int | None = None

    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=pythonpath_env(root),
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        ok = completed.returncode == 0
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timeout = True
        returncode = None
        stdout = _decode_timeout_output(exc.stdout)
        stderr = _decode_timeout_output(exc.stderr)
        error = f"Timeout despues de {TIMEOUT_SECONDS} segundos"
    except Exception as exc:  # noqa: BLE001 - this diagnostic tool must continue.
        error = repr(exc)

    duration = time.perf_counter() - start
    results.append(
        {
            "name": name,
            "duration": duration,
            "ok": ok,
            "timeout": timeout,
            "returncode": returncode,
        }
    )

    write_line(handle, f"Duracion: {duration:.3f} s")
    write_line(handle, f"Termino correctamente: {'si' if ok else 'no'}")
    write_line(handle, f"Timeout: {'si' if timeout else 'no'}")
    write_line(handle, f"Codigo de salida: {returncode}")
    if error:
        write_line(handle, "Error:")
        write_line(handle, error)
    write_block(handle, "STDOUT", stdout)
    write_block(handle, "STDERR / WARNINGS", stderr)
    write_line(handle)


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def write_block(handle, title: str, content: str) -> None:
    write_line(handle, f"{title}:")
    if content:
        handle.write(content)
        if not content.endswith("\n"):
            handle.write("\n")
    else:
        handle.write("(vacio)\n")
    handle.flush()


def run_summary(handle, results: Iterable[dict[str, object]]) -> None:
    write_line(handle, "== RESUMEN ORDENADO POR DURACION ==")
    ordered = sorted(results, key=lambda item: float(item["duration"]), reverse=True)
    for item in ordered:
        status = "OK" if item["ok"] else "FALLO"
        timeout = " timeout" if item["timeout"] else ""
        write_line(handle, f"{float(item['duration']):8.3f} s | {status}{timeout} | {item['name']}")


def snippet_full_startup() -> str:
    return r'''
import argparse
from pathlib import Path
import tempfile
import time

class FirstFrameShown(RuntimeError):
    pass

start = time.perf_counter()

import cv2
from src.recognition import predict_live
from src.config.runtime import default_config_path, load_runtime_config, apply_arg_overrides

original_imshow = cv2.imshow
original_imwrite = cv2.imwrite
original_wait_key = cv2.waitKey
first_frame_at = None

def patched_imwrite(filename, frame):
    return True

def patched_imshow(name, frame):
    global first_frame_at
    original_imshow(name, frame)
    original_wait_key(1)
    first_frame_at = time.perf_counter()
    print(f"FIRST_FRAME_SECONDS={first_frame_at - start:.6f}", flush=True)
    raise FirstFrameShown()

cv2.imshow = patched_imshow
cv2.imwrite = patched_imwrite

args = argparse.Namespace(
    input_npy=None,
    record_seconds=None,
    stride=None,
    min_confidence=None,
    save_clip=None,
    config=False,
    no_prototypes=False,
)
root = Path.cwd()
config = apply_arg_overrides(load_runtime_config(default_config_path(root)), args)

with tempfile.TemporaryDirectory() as tmpdir:
    bridge_dir = Path(tmpdir)
    try:
        predict_live.run_camera(
            args,
            config,
            bridge_dir / "output.txt",
            bridge_dir / "frame.jpg",
            bridge_dir / "debug_response.txt",
        )
    except FirstFrameShown:
        pass
    finally:
        cv2.imshow = original_imshow
        cv2.imwrite = original_imwrite
        cv2.waitKey = original_wait_key
        cv2.destroyAllWindows()
'''


def snippet_import(module_name: str) -> str:
    return f'''
import time
start = time.perf_counter()
import {module_name}
print("{module_name} import seconds", f"{{time.perf_counter() - start:.6f}}")
'''


def snippet_load_model() -> str:
    return r'''
import time
from src.recognition.model import load_sign_model
start = time.perf_counter()
model = load_sign_model()
print("load_sign_model seconds", f"{time.perf_counter() - start:.6f}")
print("input_shape", model.input_shape)
'''


def snippet_build_prototypes() -> str:
    return r'''
import time
from src.recognition.continuous import PrototypeLibrary
start = time.perf_counter()
prototypes = PrototypeLibrary.from_dataset()
print("PrototypeLibrary.from_dataset seconds", f"{time.perf_counter() - start:.6f}")
print("labels", len(prototypes.prototypes))
print("static_labels", len(prototypes.static_prototypes))
'''


def snippet_hand_tracker() -> str:
    return r'''
import time
from src.config.constants import MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from src.vision.hand_tracker import HandTracker
start = time.perf_counter()
tracker = HandTracker(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
tracker.close()
print("HandTracker create/close seconds", f"{time.perf_counter() - start:.6f}")
'''


def snippet_camera_first_frame() -> str:
    return r'''
import time
import cv2
from src.config.constants import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH
start = time.perf_counter()
cap = cv2.VideoCapture(CAMERA_INDEX)
try:
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    success, frame = cap.read()
    if not success:
        raise RuntimeError("No se pudo leer el primer frame.")
    print("camera first frame seconds", f"{time.perf_counter() - start:.6f}")
    print("frame shape", getattr(frame, "shape", None))
finally:
    cap.release()
    cv2.destroyAllWindows()
'''


def snippet_create_session() -> str:
    return r'''
import argparse
from pathlib import Path
import tempfile
import time
from src.config.constants import MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from src.vision.hand_tracker import HandTracker
from src.recognition.session import LiveRecognitionSession
from src.config.runtime import DEFAULT_FPS, default_config_path, load_runtime_config, apply_arg_overrides

args = argparse.Namespace(
    input_npy=None,
    record_seconds=None,
    stride=None,
    min_confidence=None,
    save_clip=None,
    config=False,
    no_prototypes=False,
)
root = Path.cwd()
config = apply_arg_overrides(load_runtime_config(default_config_path(root)), args)
tracker = HandTracker(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
start = time.perf_counter()
with tempfile.TemporaryDirectory() as tmpdir:
    bridge_dir = Path(tmpdir)
    try:
        session = LiveRecognitionSession.create(
            args=args,
            config=config,
            output_text_path=bridge_dir / "output.txt",
            output_frame_path=bridge_dir / "frame.jpg",
            debug_response_path=bridge_dir / "debug_response.txt",
            tracker=tracker,
            fps=DEFAULT_FPS,
            project_root=root,
            result_printer=None,
        )
        print("LiveRecognitionSession.create seconds", f"{time.perf_counter() - start:.6f}")
        print("recorder_state", session.recorder.state.value)
    finally:
        tracker.close()
'''


def python_command(executable: str, code: str) -> list[str]:
    return [executable, "-c", textwrap.dedent(code).strip()]


def main() -> None:
    root = project_root()
    output_path = result_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    with output_path.open("w", encoding="utf-8") as handle:
        write_header(handle)
        write_line(handle, f"Archivo de resultados: {output_path}")
        write_line(handle, f"Proyecto: {root}")
        write_line(handle, f"Python: {sys.executable}")
        write_line(handle, f"Timeout por prueba: {TIMEOUT_SECONDS} s")
        write_line(handle)

        run_subprocess(
            "Arranque real de predict_live hasta primer cv2.imshow",
            python_command(sys.executable, snippet_full_startup()),
            handle,
            root,
            results,
        )

        tests = [
            ("Importar TensorFlow", python_command(sys.executable, snippet_import("tensorflow"))),
            ("Importar MediaPipe", python_command(sys.executable, snippet_import("mediapipe"))),
            ("Importar predict_live", python_command(sys.executable, snippet_import("src.recognition.predict_live"))),
            ("Cargar modelo .keras", python_command(sys.executable, snippet_load_model())),
            ("Construir PrototypeLibrary.from_dataset()", python_command(sys.executable, snippet_build_prototypes())),
            ("Crear y cerrar HandTracker", python_command(sys.executable, snippet_hand_tracker())),
            ("Abrir camara y obtener primer frame", python_command(sys.executable, snippet_camera_first_frame())),
            ("Crear componentes LiveRecognitionSession", python_command(sys.executable, snippet_create_session())),
            (
                'Importtime de "import src.recognition.predict_live"',
                [sys.executable, "-X", "importtime", "-c", "import src.recognition.predict_live"],
            ),
        ]

        for name, command in tests:
            run_subprocess(name, command, handle, root, results)

        run_summary(handle, results)

    print(f"[startup] resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
