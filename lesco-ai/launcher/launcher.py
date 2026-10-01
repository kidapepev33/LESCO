#!/usr/bin/env python3
"""Lanzador local de Prisma para Windows/Linux y navegadores Chromium."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


LAUNCHER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAUNCHER_DIR.parent
CONFIG_PATH = LAUNCHER_DIR / "config.json"
ENTRY_POINT = PROJECT_ROOT / "prisma.py"
HEALTH_IDENTITY = {"service": "prisma", "status": "ok"}
BROWSER_TARGET_TIMEOUT_SECONDS = 15.0
BROWSER_TARGET_POLL_SECONDS = 0.25
WINDOWS = os.name == "nt"


class LauncherError(RuntimeError):
    """Error comprensible para el usuario del lanzador."""


class BrowserSession:
    """Chrome app window tracked through its dedicated DevTools endpoint."""

    def __init__(self, process: subprocess.Popen, profile_dir: Path, base_url: str) -> None:
        self.process = process
        self.profile_dir = profile_dir
        self.base_url = base_url.rstrip("/")

    def _debug_port(self) -> int | None:
        port_file = self.profile_dir / "DevToolsActivePort"
        try:
            first_line = port_file.read_text(encoding="utf-8").splitlines()[0]
            return int(first_line)
        except (OSError, UnicodeError, ValueError, IndexError):
            return None

    def _prisma_window_is_open(self) -> bool | None:
        port = self._debug_port()
        if port is None:
            return None
        try:
            with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=0.8) as response:
                targets = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, UnicodeError, json.JSONDecodeError):
            return None

        expected = urlparse(self.base_url)
        for target in targets:
            if target.get("type") != "page":
                continue
            current = urlparse(str(target.get("url", "")))
            if current.scheme == expected.scheme and current.netloc == expected.netloc:
                return True
        return False

    def wait_until_closed(self) -> None:
        """Wait for the actual Prisma page target, including Chrome hand-off cases."""
        deadline = time.monotonic() + BROWSER_TARGET_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            state = self._prisma_window_is_open()
            if state is True:
                break
            if self.process.poll() is not None and state is None:
                raise LauncherError(
                    "Chrome terminó sin exponer la ventana de Prisma para supervisar su cierre."
                )
            time.sleep(BROWSER_TARGET_POLL_SECONDS)
        else:
            raise LauncherError("No se pudo confirmar la apertura de la ventana de Prisma.")

        unavailable_since: float | None = None
        while True:
            state = self._prisma_window_is_open()
            if state is False:
                return
            if state is None:
                if unavailable_since is None:
                    unavailable_since = time.monotonic()
                elif time.monotonic() - unavailable_since >= 2.0:
                    return
            else:
                unavailable_since = None
            time.sleep(BROWSER_TARGET_POLL_SECONDS)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LauncherError(f"No se pudo leer la configuración {path}: {exc}") from exc

    required = {"auto_start_backend", "stop_backend_on_exit"}
    missing = sorted(required.difference(config))
    if missing:
        raise LauncherError("Faltan opciones en config.json: " + ", ".join(missing))
    return config


def health_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/health"


def prisma_is_ready(base_url: str, timeout: float = 0.8) -> bool:
    request = Request(health_url(base_url), headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, UnicodeError, json.JSONDecodeError):
        return False
    return payload == HEALTH_IDENTITY


def configured_port_is_open(base_url: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def project_python(config: dict[str, Any]) -> Path:
    configured = config.get("python_executable")
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    virtual_python = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if WINDOWS
        else PROJECT_ROOT / ".venv" / "bin" / "python"
    )
    return virtual_python if virtual_python.is_file() else Path(sys.executable)


def new_process_group_options() -> dict[str, Any]:
    """Return flags for an independently controllable child process."""
    if WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)}
    return {"start_new_session": True}


def start_backend(config: dict[str, Any]) -> tuple[subprocess.Popen, Any]:
    python = project_python(config)
    if not python.is_file():
        raise LauncherError(f"No se encontró el intérprete de Python: {python}")
    if not ENTRY_POINT.is_file():
        raise LauncherError(f"No se encontró el punto de entrada: {ENTRY_POINT}")

    log_dir = LAUNCHER_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (log_dir / "backend.log").open("a", encoding="utf-8")
    log_handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Inicio solicitado por launcher.\n")
    log_handle.flush()

    try:
        process = subprocess.Popen(
            [str(python), "-u", str(ENTRY_POINT)],
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **new_process_group_options(),
        )
    except Exception:
        log_handle.close()
        raise
    return process, log_handle


def stop_owned_backend(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if WINDOWS:
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
        else:
            os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def wait_for_backend(process: subprocess.Popen, base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if prisma_is_ready(base_url):
            return
        return_code = process.poll()
        if return_code is not None:
            raise LauncherError(
                f"Prisma terminó antes de iniciar Flask (código {return_code}). "
                "Revisa launcher/logs/backend.log."
            )
        time.sleep(0.25)
    raise LauncherError(
        f"Flask no estuvo listo después de {timeout:g} segundos. "
        "Revisa launcher/logs/backend.log."
    )


def find_browser(config: dict[str, Any]) -> str:
    configured = config.get("browser_executable")
    if configured:
        resolved = shutil.which(configured) or (configured if Path(configured).is_file() else None)
        if resolved:
            return str(resolved)

    for candidate in config.get("browser_candidates", []):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    if WINDOWS:
        for executable in ("chrome.exe", "msedge.exe"):
            resolved = shutil.which(executable)
            if resolved:
                return resolved
        roots = (
            os.environ.get("PROGRAMFILES"),
            os.environ.get("PROGRAMFILES(X86)"),
            os.environ.get("LOCALAPPDATA"),
        )
        suffixes = (
            Path("Google/Chrome/Application/chrome.exe"),
            Path("Microsoft/Edge/Application/msedge.exe"),
        )
        for root in roots:
            if root:
                for suffix in suffixes:
                    candidate = Path(root) / suffix
                    if candidate.is_file():
                        return str(candidate)
    raise LauncherError("No se encontró Chrome, Chromium o Edge. Ajusta browser_executable en config.json.")


def browser_profile_dir(config: dict[str, Any]) -> Path:
    configured = str(config.get("browser_profile_dir", "")).strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    if WINDOWS:
        local_app_data = os.environ.get("LOCALAPPDATA")
        state_home = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return state_home / "Prisma" / "browser-profile"
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    return state_home / "prisma" / "browser-profile"


def open_prisma_window(config: dict[str, Any]) -> tuple[BrowserSession, str]:
    browser = find_browser(config)
    profile_dir = browser_profile_dir(config)
    profile_dir.mkdir(parents=True, exist_ok=True)
    app_id = str(config.get("installed_app_id", "")).strip()
    command = [
        browser,
        f"--user-data-dir={profile_dir}",
        "--remote-debugging-port=0",
        "--no-first-run",
        "--disable-background-mode",
    ]
    if bool(config.get("fullscreen", False)):
        command.append("--start-fullscreen")
    if app_id:
        command.append(f"--app-id={app_id}")
        mode = "PWA instalada configurada mediante installed_app_id"
    else:
        command.append(f"--app={config['base_url']}")
        mode = "ventana Chrome en modo aplicación (respaldo; no verifica instalación PWA)"
    process = subprocess.Popen(command, **new_process_group_options())
    return BrowserSession(process, profile_dir, str(config["base_url"])), mode


@contextmanager
def launcher_lock():
    """Hold one launcher instance per project on Windows and POSIX."""
    digest = hashlib.sha256(str(PROJECT_ROOT).encode("utf-8")).hexdigest()[:12]
    path = Path(tempfile.gettempdir()) / f"prisma-launcher-{digest}.lock"
    handle = path.open("a+b")
    try:
        if WINDOWS:
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        yield handle
    finally:
        if WINDOWS:
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def run() -> int:
    owned_backend: subprocess.Popen | None = None
    backend_log = None
    browser_session: BrowserSession | None = None

    try:
        config = load_config()
        base_url = str(config.get("base_url", "http://127.0.0.1:5000")).rstrip("/")
        config["base_url"] = base_url

        with launcher_lock():
            ready = prisma_is_ready(base_url)
            if not ready and configured_port_is_open(base_url):
                raise LauncherError(
                    f"El puerto de {base_url} está ocupado, pero no responde como Prisma. "
                    "No se iniciará ni abrirá ese servicio."
                )

            if not ready and bool(config["auto_start_backend"]):
                print("[PRISMA LAUNCHER] Iniciando prisma.py…", flush=True)
                owned_backend, backend_log = start_backend(config)
                try:
                    wait_for_backend(
                        owned_backend,
                        base_url,
                        float(config.get("backend_start_timeout_seconds", 45)),
                    )
                except Exception:
                    stop_owned_backend(owned_backend)
                    raise
                ready = True
            elif not ready:
                print(
                    "[PRISMA LAUNCHER] El backend no está disponible y auto_start_backend está deshabilitado. "
                    "Se abrirá la interfaz; los datos en vivo mostrarán desconexión.",
                    file=sys.stderr,
                    flush=True,
                )

            browser_session, mode = open_prisma_window(config)
            print(f"[PRISMA LAUNCHER] Abriendo {config.get('app_name', 'Prisma')}: {mode}.", flush=True)

        if owned_backend is not None and bool(config["stop_backend_on_exit"]):
            print("[PRISMA LAUNCHER] El backend se cerrará al salir de esta ventana.", flush=True)
            try:
                browser_session.wait_until_closed()
            finally:
                stop_owned_backend(owned_backend)
        return 0
    except LauncherError as exc:
        print(f"[PRISMA LAUNCHER] Error: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        if owned_backend is not None and bool(load_config().get("stop_backend_on_exit", False)):
            stop_owned_backend(owned_backend)
        return 130
    finally:
        if backend_log is not None:
            backend_log.close()


if __name__ == "__main__":
    raise SystemExit(run())
