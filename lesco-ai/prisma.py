"""Punto de entrada único para ejecutar los servicios locales de Prisma."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
GRACEFUL_TIMEOUT_SECONDS = 5.0
TERMINATE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class Component:
    """Describe un proceso que forma parte de la ejecución normal de Prisma."""

    name: str
    script: Path


COMPONENTS = (
    Component("Web Flask", PROJECT_ROOT / "web" / "web_api.py"),
    Component("Puente de videos", PROJECT_ROOT / "src" / "sign_video_bridge.py"),
    Component("Reconocimiento en vivo", PROJECT_ROOT / "src" / "predict_live.py"),
)


def component_command(component: Component) -> list[str]:
    """Construye el comando usando el mismo intérprete que inició Prisma."""
    return [sys.executable, "-u", str(component.script)]


def validate_components(components: Sequence[Component] = COMPONENTS) -> None:
    """Falla antes de abrir procesos si falta algún punto de entrada."""
    missing = [str(component.script) for component in components if not component.script.is_file()]
    if missing:
        raise FileNotFoundError("No se encontraron componentes de Prisma: " + ", ".join(missing))


def start_components(
    components: Sequence[Component] = COMPONENTS,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> list[tuple[Component, subprocess.Popen]]:
    """Inicia cada componente en una sesión propia para controlar su cierre."""
    validate_components(components)
    processes: list[tuple[Component, subprocess.Popen]] = []
    try:
        for component in components:
            command = component_command(component)
            print(f"[PRISMA] Iniciando {component.name}: {' '.join(command)}", flush=True)
            process = popen_factory(
                command,
                cwd=PROJECT_ROOT,
                start_new_session=(os.name != "nt"),
            )
            processes.append((component, process))
    except Exception:
        stop_components(processes)
        raise
    return processes


def request_graceful_stop(process: subprocess.Popen) -> None:
    """Solicita cierre con Ctrl+C para ejecutar los bloques finally de cada hijo."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(process.pid, signal.SIGINT)


def stop_components(processes: Sequence[tuple[Component, subprocess.Popen]]) -> None:
    """Cierra hijos ordenadamente y escala a terminate/kill solo si es necesario."""
    active = [(component, process) for component, process in processes if process.poll() is None]
    if not active:
        return

    print("[PRISMA] Cerrando componentes…", flush=True)
    for _, process in active:
        try:
            request_graceful_stop(process)
        except ProcessLookupError:
            pass

    graceful_deadline = time.monotonic() + GRACEFUL_TIMEOUT_SECONDS
    for component, process in active:
        remaining = max(0.0, graceful_deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[PRISMA] {component.name} no respondió; enviando terminate.", flush=True)

    stubborn = [(component, process) for component, process in active if process.poll() is None]
    for _, process in stubborn:
        process.terminate()

    terminate_deadline = time.monotonic() + TERMINATE_TIMEOUT_SECONDS
    for component, process in stubborn:
        remaining = max(0.0, terminate_deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[PRISMA] Forzando cierre de {component.name}.", flush=True)
            process.kill()

    for _, process in stubborn:
        if process.poll() is None:
            process.wait()


def run() -> int:
    """Mantiene Prisma activo mientras sus tres componentes estén saludables."""
    processes: list[tuple[Component, subprocess.Popen]] = []
    exit_code = 0
    try:
        processes = start_components()
        print("[PRISMA] Listo: http://127.0.0.1:5000", flush=True)
        print("[PRISMA] Presiona Ctrl+C para cerrar todos los componentes.", flush=True)

        while True:
            for component, process in processes:
                return_code = process.poll()
                if return_code is not None:
                    print(
                        f"[PRISMA] {component.name} terminó con código {return_code}; "
                        "se cerrarán los demás componentes.",
                        flush=True,
                    )
                    return 0 if return_code == 0 else 1
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[PRISMA] Cierre solicitado.", flush=True)
    except Exception as exc:
        exit_code = 1
        print(f"[PRISMA] No se pudo iniciar: {exc}", file=sys.stderr, flush=True)
    finally:
        stop_components(processes)
        print("[PRISMA] Todos los componentes finalizaron.", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
