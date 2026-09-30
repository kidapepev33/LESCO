"""Mide el arranque real de Prisma sin ejecutar predicciones.

Lanza ``prisma.py``, observa su árbol desde ``/proc`` y espera señales externas
de disponibilidad. No importa módulos de la aplicación ni instrumenta el bucle
de reconocimiento, para alterar lo menos posible la medición.

Uso: ``.venv/bin/python tools/measure_startup.py``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import datetime as dt
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

DEFAULT_INTERVAL_SECONDS = 0.20
DEFAULT_READY_TIMEOUT_SECONDS = 120.0
DEFAULT_STEADY_SECONDS = 10.0
BRIDGE_STABLE_SECONDS = 1.0
WEB_URL = "http://127.0.0.1:5000/resultado"


@dataclass
class ProcessReading:
    pid: int
    ppid: int
    cpu_seconds: float
    rss_bytes: int
    command: str


@dataclass
class ComponentStats:
    startup_peak_rss_bytes: int = 0
    steady_peak_rss_bytes: int = 0
    cpu_at_start: float | None = None
    cpu_at_component_ready: float | None = None
    cpu_at_all_ready: float | None = None
    cpu_at_end: float | None = None
    readiness_seconds: float | None = None


@dataclass
class Measurement:
    started_wall: str
    interval_seconds: float
    steady_seconds_requested: float
    milestones: dict[str, float] = field(default_factory=dict)
    components: dict[str, ComponentStats] = field(
        default_factory=lambda: {
            name: ComponentStats()
            for name in ("prisma", "web", "video_bridge", "recognizer", "other_children", "total_tree")
        }
    )
    tree_peak_rss_startup: int = 0
    tree_peak_rss_steady: int = 0
    steady_rss_sum: int = 0
    steady_samples: int = 0
    samples: int = 0
    ready: bool = False
    failure: str | None = None
    prisma_returncode: int | None = None
    log_tail: list[str] = field(default_factory=list)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_path() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"startup_results_{stamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mide tiempo, CPU y RAM del arranque de prisma.py.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_READY_TIMEOUT_SECONDS)
    parser.add_argument("--steady-seconds", type=float, default=DEFAULT_STEADY_SECONDS)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_process(pid: int) -> ProcessReading | None:
    """Lee una instantánea barata de un proceso Linux desde /proc."""
    try:
        process_dir = Path("/proc") / str(pid)
        stat = (process_dir / "stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
        return ProcessReading(
            pid=pid,
            ppid=int(fields[1]),
            cpu_seconds=(int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
            rss_bytes=max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE"),
            command=command,
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
        return None


def process_tree(root_pid: int) -> list[ProcessReading]:
    try:
        pids = [int(entry.name) for entry in Path("/proc").iterdir() if entry.name.isdigit()]
    except (FileNotFoundError, PermissionError) as exc:
        raise RuntimeError("Se requiere /proc para medir procesos hijos.") from exc
    readings = {reading.pid: reading for pid in pids if (reading := read_process(pid)) is not None}
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for reading in readings.values():
            if reading.ppid in selected and reading.pid not in selected:
                selected.add(reading.pid)
                changed = True
    return [readings[pid] for pid in selected if pid in readings]


def direct_component(reading: ProcessReading, root_pid: int) -> str | None:
    if reading.pid == root_pid:
        return "prisma"
    command = reading.command.replace("\\", "/")
    if "web/web_api.py" in command:
        return "web"
    if "src/sign_video_bridge.py" in command:
        return "video_bridge"
    if "src/predict_live.py" in command:
        return "recognizer"
    return None


def grouped_totals(tree: list[ProcessReading], root_pid: int) -> dict[str, tuple[float, int]]:
    names = ("prisma", "web", "video_bridge", "recognizer", "other_children")
    totals: dict[str, list[float | int]] = {name: [0.0, 0] for name in names}
    by_pid = {reading.pid: reading for reading in tree}

    def owner(reading: ProcessReading) -> str:
        current = reading
        visited: set[int] = set()
        while current.pid not in visited:
            visited.add(current.pid)
            component = direct_component(current, root_pid)
            if component is not None:
                return component
            parent = by_pid.get(current.ppid)
            if parent is None:
                break
            current = parent
        return "other_children"

    for reading in tree:
        bucket = totals[owner(reading)]
        bucket[0] = float(bucket[0]) + reading.cpu_seconds
        bucket[1] = int(bucket[1]) + reading.rss_bytes
    result = {name: (float(value[0]), int(value[1])) for name, value in totals.items()}
    result["total_tree"] = (
        sum(reading.cpu_seconds for reading in tree),
        sum(reading.rss_bytes for reading in tree),
    )
    return result


def frame_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def web_is_ready() -> bool:
    try:
        with urlopen(WEB_URL, timeout=0.10) as response:  # noqa: S310 - URL local fija.
            return response.status == 200
    except (URLError, TimeoutError, OSError):
        return False


def drain_output(stream, selector: selectors.BaseSelector, lines: list[str]) -> None:
    for key, _ in selector.select(timeout=0):
        line = key.fileobj.readline()
        if line:
            clean = line.rstrip()
            print(f"[prisma] {clean}")
            lines.append(clean)
            del lines[:-80]


def drain_remaining_output(stream, lines: list[str]) -> None:
    """Recoge el diagnóstico que quedó en el pipe después de terminar Prisma."""
    for line in stream.read().splitlines():
        print(f"[prisma] {line}")
        lines.append(line)
        del lines[:-80]


def stop_prisma(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def update_stats(measurement: Measurement, totals: dict[str, tuple[float, int]], phase: str) -> None:
    measurement.samples += 1
    for name, (cpu_seconds, rss_bytes) in totals.items():
        stats = measurement.components[name]
        # Un total cero también representa un proceso que ya desapareció. No
        # debe borrar su última lectura acumulada.
        if cpu_seconds == 0.0 and rss_bytes == 0:
            continue
        if stats.cpu_at_start is None:
            stats.cpu_at_start = 0.0
        stats.cpu_at_end = max(stats.cpu_at_end or 0.0, cpu_seconds)
        if phase == "startup":
            stats.startup_peak_rss_bytes = max(stats.startup_peak_rss_bytes, rss_bytes)
        else:
            stats.steady_peak_rss_bytes = max(stats.steady_peak_rss_bytes, rss_bytes)
    tree_rss = totals["total_tree"][1]
    if phase == "startup":
        measurement.tree_peak_rss_startup = max(measurement.tree_peak_rss_startup, tree_rss)
    else:
        measurement.tree_peak_rss_steady = max(measurement.tree_peak_rss_steady, tree_rss)
        measurement.steady_rss_sum += tree_rss
        measurement.steady_samples += 1


def mark_ready(measurement: Measurement, name: str, elapsed: float, cpu_seconds: float) -> None:
    measurement.milestones[name] = elapsed
    stats = measurement.components[name]
    stats.readiness_seconds = elapsed
    stats.cpu_at_component_ready = cpu_seconds
    print(f"[medidor] {name} listo a los {elapsed:.3f} s")


def serialise(measurement: Measurement, command: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "metadata": {
            "started_at": measurement.started_wall,
            "command": command,
            "sample_interval_seconds": measurement.interval_seconds,
            "steady_seconds_requested": measurement.steady_seconds_requested,
            "cpu_metric": "segundos de CPU acumulados (usuario + sistema)",
            "ram_metric": "suma de RSS; memoria compartida puede contarse más de una vez",
        },
        "status": {
            "all_components_ready": measurement.ready,
            "failure": measurement.failure,
            "prisma_returncode": measurement.prisma_returncode,
        },
        "milestones_seconds": measurement.milestones,
        "tree": {
            "startup_peak_rss_mib": measurement.tree_peak_rss_startup / 1024**2,
            "steady_peak_rss_mib": measurement.tree_peak_rss_steady / 1024**2,
            "steady_average_rss_mib": (
                measurement.steady_rss_sum / measurement.steady_samples / 1024**2
                if measurement.steady_samples else None
            ),
            "samples": measurement.samples,
        },
        "components": {},
        "readiness_definition": {
            "web": f"HTTP 200 de {WEB_URL}",
            "video_bridge": f"proceso vivo durante {BRIDGE_STABLE_SECONDS:.1f} s (no expone healthcheck)",
            "recognizer": "frame.jpg publicado o actualizado después de lanzar prisma.py",
            "all": "las tres condiciones anteriores",
        },
        "prisma_log_tail": measurement.log_tail,
    }
    for name, stats in measurement.components.items():
        data["components"][name] = {
            "ready_seconds": stats.readiness_seconds,
            "cpu_seconds_at_component_ready": stats.cpu_at_component_ready,
            "startup_cpu_seconds": (
                None if stats.cpu_at_all_ready is None else max(0.0, stats.cpu_at_all_ready)
            ),
            "steady_cpu_seconds": (
                None if stats.cpu_at_all_ready is None or stats.cpu_at_end is None
                else max(0.0, stats.cpu_at_end - stats.cpu_at_all_ready)
            ),
            "total_cpu_seconds": (
                None if stats.cpu_at_end is None else max(0.0, stats.cpu_at_end)
            ),
            "startup_peak_rss_mib": stats.startup_peak_rss_bytes / 1024**2,
            "steady_peak_rss_mib": stats.steady_peak_rss_bytes / 1024**2,
        }
    return data


def main() -> int:
    args = parse_args()
    if args.interval < 0.05 or args.timeout <= 0 or args.steady_seconds < 0:
        raise SystemExit("--interval debe ser >= 0.05; --timeout > 0; --steady-seconds >= 0")
    if not Path("/proc").is_dir():
        raise SystemExit("Este medidor requiere Linux y /proc para incluir procesos hijos.")

    root = project_root()
    output = (args.output or default_output_path()).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_path = root / "godot_bridge" / "frame.jpg"
    old_frame = frame_signature(frame_path)
    command = [sys.executable, "-u", str(root / "prisma.py")]
    measurement = Measurement(
        started_wall=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        interval_seconds=args.interval,
        steady_seconds_requested=args.steady_seconds,
    )

    print("[medidor] Ejecutando:", " ".join(command))
    started = time.perf_counter()
    process = subprocess.Popen(
        command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    bridge_seen_at: float | None = None
    ready_at: float | None = None

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - started
            drain_output(process.stdout, selector, measurement.log_tail)
            tree = process_tree(process.pid)
            totals = grouped_totals(tree, process.pid)
            update_stats(measurement, totals, "steady" if ready_at is not None else "startup")
            commands = "\n".join(reading.command for reading in tree)

            if "web" not in measurement.milestones and "web/web_api.py" in commands and web_is_ready():
                mark_ready(measurement, "web", elapsed, totals["web"][0])

            bridge_alive = "src/sign_video_bridge.py" in commands
            if bridge_alive and bridge_seen_at is None:
                bridge_seen_at = now
            elif not bridge_alive:
                bridge_seen_at = None
            if ("video_bridge" not in measurement.milestones and bridge_seen_at is not None
                    and now - bridge_seen_at >= BRIDGE_STABLE_SECONDS):
                mark_ready(measurement, "video_bridge", elapsed, totals["video_bridge"][0])

            current_frame = frame_signature(frame_path)
            if ("recognizer" not in measurement.milestones and "src/predict_live.py" in commands
                    and current_frame is not None and current_frame != old_frame):
                mark_ready(measurement, "recognizer", elapsed, totals["recognizer"][0])

            if ready_at is None and len(measurement.milestones) == 3:
                ready_at = now
                measurement.ready = True
                measurement.milestones["all_components"] = elapsed
                for name in measurement.components:
                    stats = measurement.components[name]
                    stats.cpu_at_all_ready = totals[name][0]
                    if name in {"total_tree", "prisma", "other_children"}:
                        stats.readiness_seconds = elapsed
                print(f"[medidor] Prisma completamente listo a los {elapsed:.3f} s")

            returncode = process.poll()
            if returncode is not None:
                measurement.prisma_returncode = returncode
                if not measurement.ready:
                    measurement.failure = f"prisma.py terminó antes de estar listo (código {returncode})"
                break
            if ready_at is None and elapsed >= args.timeout:
                missing = sorted({"web", "video_bridge", "recognizer"} - measurement.milestones.keys())
                measurement.failure = "timeout esperando: " + ", ".join(missing)
                break
            if ready_at is not None and now - ready_at >= args.steady_seconds:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        measurement.failure = "medición interrumpida por el usuario"
    finally:
        stop_prisma(process)
        measurement.prisma_returncode = process.returncode
        drain_output(process.stdout, selector, measurement.log_tail)
        drain_remaining_output(process.stdout, measurement.log_tail)
        selector.close()

    output.write_text(json.dumps(serialise(measurement, command), ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"[medidor] Resultados: {output}")
    if measurement.failure:
        print(f"[medidor] Medición incompleta: {measurement.failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
