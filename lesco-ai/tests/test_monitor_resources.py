"""Tests for the standalone running-Prisma resource monitor."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.measure_startup import ProcessReading
from tools.monitor_resources import (
    MonitorState,
    ResourceStats,
    process_owners,
    monitor,
    render_report,
    sample_deltas,
    select_prisma_pid,
    write_report_atomic,
)


def reading(pid: int, ppid: int, cpu: float, rss: int, command: str) -> ProcessReading:
    return ProcessReading(pid, ppid, cpu, rss, command)


class MonitorResourcesTests(unittest.TestCase):
    def test_requires_pid_when_multiple_prisma_instances_exist(self) -> None:
        matches = [
            reading(10, 1, 0.0, 1, "python /project/prisma.py"),
            reading(20, 1, 0.0, 1, "python /project/prisma.py"),
        ]
        with patch("tools.monitor_resources.find_prisma_processes", return_value=matches):
            with self.assertRaisesRegex(ValueError, r"10, 20.*--pid"):
                select_prisma_pid(None)

    def test_explicit_pid_must_be_prisma(self) -> None:
        with patch(
            "tools.monitor_resources.read_process",
            return_value=reading(50, 1, 0.0, 1, "python other.py"),
        ):
            with self.assertRaisesRegex(ValueError, "no corresponde"):
                select_prisma_pid(50)

    def test_groups_each_process_once_and_calculates_cpu_delta(self) -> None:
        tree = [
            reading(10, 1, 2.0, 10, "python prisma.py"),
            reading(11, 10, 3.0, 20, "python web/web_api.py"),
            reading(12, 10, 5.0, 30, "python src/predict_live.py"),
            reading(13, 12, 7.0, 40, "worker"),
        ]
        self.assertEqual(process_owners(tree, 10), {10: "prisma", 11: "web", 12: "recognizer", 13: "recognizer"})

        cpu, rss, current = sample_deltas(tree, 10, {10: 1.5, 11: 2.0, 12: 4.0, 13: 6.0})

        self.assertEqual(cpu["prisma"], 0.5)
        self.assertEqual(cpu["web"], 1.0)
        self.assertEqual(cpu["recognizer"], 2.0)
        self.assertEqual(cpu["total_tree"], 3.5)
        self.assertEqual(rss["recognizer"], 70)
        self.assertEqual(rss["total_tree"], 100)
        self.assertEqual(set(current), {10, 11, 12, 13})

    def test_resource_statistics_use_elapsed_monotonic_interval(self) -> None:
        stats = ResourceStats()
        stats.update(cpu_seconds=1.5, rss_bytes=20 * 1024**2, interval=0.5)
        self.assertEqual(stats.current_cpu_percent, 300.0)
        self.assertEqual(stats.max_cpu_percent, 300.0)
        self.assertEqual(stats.cpu_seconds_observed, 1.5)

    def test_atomic_report_is_complete_and_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "debug_resources.txt"
            write_report_atomic(path, "reporte completo\n")
            temporary_files = list(path.parent.glob(".debug_resources.txt.tmp.*"))
            self.assertEqual(path.read_text(encoding="utf-8"), "reporte completo\n")
            self.assertEqual(temporary_files, [])

    def test_report_documents_metrics_and_final_status(self) -> None:
        state = MonitorState(root_pid=123, started_at=10.0, elapsed=2.0, samples=2)
        state.stats["total_tree"].update(1.0, 100 * 1024**2, 2.0)
        report = render_report(state, "FINALIZADO: Prisma terminó")
        self.assertIn("PID de Prisma: 123", report)
        self.assertIn("100% equivale a un núcleo", report)
        self.assertIn("memoria compartida puede contarse", report)
        self.assertIn("FINALIZADO: Prisma terminó", report)

    def test_monitor_writes_final_summary_when_prisma_ends(self) -> None:
        root = reading(123, 1, 1.0, 10, "python prisma.py")
        reports: list[str] = []
        with (
            patch("tools.monitor_resources.process_tree", side_effect=([root], [])),
            patch("tools.monitor_resources.time.monotonic", side_effect=(0.0, 0.0, 1.0)),
            patch("tools.monitor_resources.time.sleep"),
            patch("tools.monitor_resources.write_report_atomic", side_effect=lambda path, text: reports.append(text)),
        ):
            monitor(123, Path("unused"))

        self.assertEqual(len(reports), 2)
        self.assertIn("MONITOREANDO", reports[0])
        self.assertIn("FINALIZADO: Prisma terminó", reports[-1])

    def test_ctrl_c_stops_only_monitor_and_preserves_report(self) -> None:
        root = reading(123, 1, 1.0, 10, "python prisma.py")
        reports: list[str] = []
        with (
            patch("tools.monitor_resources.process_tree", return_value=[root]),
            patch("tools.monitor_resources.time.monotonic", side_effect=(0.0, 0.0, 0.5)),
            patch("tools.monitor_resources.time.sleep", side_effect=KeyboardInterrupt),
            patch("tools.monitor_resources.write_report_atomic", side_effect=lambda path, text: reports.append(text)),
        ):
            monitor(123, Path("unused"))

        self.assertIn("INTERRUMPIDO POR EL USUARIO", reports[-1])


if __name__ == "__main__":
    unittest.main()
