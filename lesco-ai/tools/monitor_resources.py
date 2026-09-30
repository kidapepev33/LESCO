"""Monitor a running Prisma process tree and publish lightweight diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import time

try:
    from .measure_startup import ProcessReading, direct_component, grouped_totals, process_tree, read_process
except ImportError:  # Direct execution from tools/.
    from measure_startup import ProcessReading, direct_component, grouped_totals, process_tree, read_process


INTERVAL_SECONDS = 1.0
COMPONENTS = ("prisma", "web", "video_bridge", "recognizer", "other_children", "total_tree")
DISPLAY_NAMES = {
    "prisma": "Prisma",
    "web": "Web",
    "video_bridge": "Puente de videos",
    "recognizer": "Reconocedor",
    "other_children": "Otros descendientes",
    "total_tree": "Árbol completo",
}


@dataclass
class ResourceStats:
    current_cpu_percent: float = 0.0
    cpu_seconds_observed: float = 0.0
    max_cpu_percent: float = 0.0
    current_rss_bytes: int = 0
    weighted_rss_bytes_seconds: float = 0.0
    max_rss_bytes: int = 0

    def update(self, cpu_seconds: float, rss_bytes: int, interval: float) -> None:
        self.current_cpu_percent = cpu_seconds / interval * 100.0 if interval > 0 else 0.0
        self.cpu_seconds_observed += cpu_seconds
        self.max_cpu_percent = max(self.max_cpu_percent, self.current_cpu_percent)
        self.current_rss_bytes = rss_bytes
        self.weighted_rss_bytes_seconds += rss_bytes * interval
        self.max_rss_bytes = max(self.max_rss_bytes, rss_bytes)


@dataclass
class MonitorState:
    root_pid: int
    started_at: float
    elapsed: float = 0.0
    samples: int = 0
    stats: dict[str, ResourceStats] = field(
        default_factory=lambda: {name: ResourceStats() for name in COMPONENTS}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitorea recursos de una instancia activa de Prisma.")
    parser.add_argument("--pid", type=int, help="PID explícito del proceso prisma.py")
    return parser.parse_args()


def is_prisma_process(reading: ProcessReading) -> bool:
    return any(Path(part).name == "prisma.py" for part in reading.command.split())


def find_prisma_processes() -> list[ProcessReading]:
    if not Path("/proc").is_dir():
        raise RuntimeError("Esta herramienta requiere Linux y /proc.")
    readings: list[ProcessReading] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        reading = read_process(int(entry.name))
        if reading is not None and is_prisma_process(reading):
            readings.append(reading)
    return sorted(readings, key=lambda reading: reading.pid)


def select_prisma_pid(explicit_pid: int | None) -> int:
    if explicit_pid is not None:
        reading = read_process(explicit_pid)
        if reading is None:
            raise ValueError(f"No existe un proceso accesible con PID {explicit_pid}.")
        if not is_prisma_process(reading):
            raise ValueError(f"El PID {explicit_pid} no corresponde a prisma.py.")
        return explicit_pid

    matches = find_prisma_processes()
    if not matches:
        raise ValueError("No se encontró ninguna instancia activa de prisma.py.")
    if len(matches) > 1:
        pids = ", ".join(str(reading.pid) for reading in matches)
        raise ValueError(f"Se encontraron varias instancias de prisma.py ({pids}). Usa --pid PID.")
    return matches[0].pid


def process_owners(tree: list[ProcessReading], root_pid: int) -> dict[int, str]:
    by_pid = {reading.pid: reading for reading in tree}
    owners: dict[int, str] = {}
    for reading in tree:
        current = reading
        visited: set[int] = set()
        while current.pid not in visited:
            visited.add(current.pid)
            component = direct_component(current, root_pid)
            if component is not None:
                owners[reading.pid] = component
                break
            parent = by_pid.get(current.ppid)
            if parent is None:
                owners[reading.pid] = "other_children"
                break
            current = parent
    return owners


def sample_deltas(
    tree: list[ProcessReading],
    root_pid: int,
    previous_cpu: dict[int, float],
) -> tuple[dict[str, float], dict[str, int], dict[int, float]]:
    owners = process_owners(tree, root_pid)
    cpu = {name: 0.0 for name in COMPONENTS}
    for reading in tree:
        delta = max(0.0, reading.cpu_seconds - previous_cpu.get(reading.pid, reading.cpu_seconds))
        cpu[owners[reading.pid]] += delta
        cpu["total_tree"] += delta
    rss = {name: value[1] for name, value in grouped_totals(tree, root_pid).items()}
    current_cpu = {reading.pid: reading.cpu_seconds for reading in tree}
    return cpu, rss, current_cpu


def render_report(state: MonitorState, status: str) -> str:
    lines = [
        "=== RECURSOS DE PRISMA ===",
        f"Estado del monitor: {status}",
        f"PID de Prisma: {state.root_pid}",
        f"Tiempo monitoreado: {state.elapsed:.1f} s",
        f"Muestras: {state.samples}",
        "",
        "CPU: 100% equivale a un núcleo completamente ocupado.",
        "RAM: suma de RSS; la memoria compartida puede contarse más de una vez.",
        "",
    ]
    for name in COMPONENTS:
        stats = state.stats[name]
        average_cpu = stats.cpu_seconds_observed / state.elapsed * 100.0 if state.elapsed > 0 else 0.0
        average_rss = (
            stats.weighted_rss_bytes_seconds / state.elapsed if state.elapsed > 0 else 0.0
        )
        lines.extend(
            (
                f"[{DISPLAY_NAMES[name]}]",
                f"CPU actual/promedio/máximo: {stats.current_cpu_percent:.1f}% / "
                f"{average_cpu:.1f}% / {stats.max_cpu_percent:.1f}%",
                f"RAM actual/promedio/máximo: {stats.current_rss_bytes / 1024**2:.1f} MiB / "
                f"{average_rss / 1024**2:.1f} MiB / {stats.max_rss_bytes / 1024**2:.1f} MiB",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(contents, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def monitor(root_pid: int, output_path: Path) -> None:
    started = time.monotonic()
    state = MonitorState(root_pid=root_pid, started_at=started)
    tree = process_tree(root_pid)
    if not any(reading.pid == root_pid for reading in tree):
        raise ValueError(f"La instancia de Prisma con PID {root_pid} ya terminó.")
    previous_cpu = {reading.pid: reading.cpu_seconds for reading in tree}
    previous_time = started
    deadline = started + INTERVAL_SECONDS
    write_report_atomic(output_path, render_report(state, "MONITOREANDO"))

    try:
        while True:
            time.sleep(max(0.0, deadline - time.monotonic()))
            now = time.monotonic()
            tree = process_tree(root_pid)
            if not any(reading.pid == root_pid for reading in tree):
                state.elapsed = now - started
                for stats in state.stats.values():
                    stats.current_cpu_percent = 0.0
                    stats.current_rss_bytes = 0
                write_report_atomic(output_path, render_report(state, "FINALIZADO: Prisma terminó"))
                return

            interval = now - previous_time
            cpu, rss, previous_cpu = sample_deltas(tree, root_pid, previous_cpu)
            for name in COMPONENTS:
                state.stats[name].update(cpu[name], rss[name], interval)
            state.samples += 1
            state.elapsed = now - started
            write_report_atomic(output_path, render_report(state, "MONITOREANDO"))
            previous_time = now
            deadline += INTERVAL_SECONDS
    except KeyboardInterrupt:
        state.elapsed = time.monotonic() - started
        write_report_atomic(output_path, render_report(state, "INTERRUMPIDO POR EL USUARIO"))


def main() -> int:
    args = parse_args()
    try:
        root_pid = select_prisma_pid(args.pid)
        output = Path(__file__).resolve().parents[1] / "godot_bridge" / "debug_resources.txt"
        print(f"[recursos] Monitoreando prisma.py PID {root_pid}")
        print(f"[recursos] Reporte: {output}")
        monitor(root_pid, output)
    except (ValueError, RuntimeError) as exc:
        print(f"[recursos] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
