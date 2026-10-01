from pathlib import Path
from contextlib import nullcontext
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from launcher import launcher  # noqa: E402


class LocalLauncherTests(unittest.TestCase):
    def test_start_and_stop_preferences_are_independent_booleans(self) -> None:
        config = launcher.load_config()
        self.assertTrue(config["auto_start_backend"])
        self.assertIsInstance(config["stop_backend_on_exit"], bool)
        self.assertEqual(config["base_url"], "http://127.0.0.1:5000")

    def test_health_requires_prisma_identity(self) -> None:
        good = Mock()
        good.status = 200
        good.read.return_value = json.dumps(launcher.HEALTH_IDENTITY).encode()
        good.__enter__ = Mock(return_value=good)
        good.__exit__ = Mock(return_value=False)

        wrong = Mock()
        wrong.status = 200
        wrong.read.return_value = b'{"service":"other","status":"ok"}'
        wrong.__enter__ = Mock(return_value=wrong)
        wrong.__exit__ = Mock(return_value=False)

        with patch.object(launcher, "urlopen", return_value=good):
            self.assertTrue(launcher.prisma_is_ready("http://127.0.0.1:5000"))
        with patch.object(launcher, "urlopen", return_value=wrong):
            self.assertFalse(launcher.prisma_is_ready("http://127.0.0.1:5000"))

    def test_browser_fallback_is_not_reported_as_installed_pwa(self) -> None:
        process = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "base_url": "http://127.0.0.1:5000",
                "installed_app_id": "",
                "browser_candidates": ["google-chrome"],
                "browser_profile_dir": tmpdir,
                "fullscreen": True,
            }
            with patch.object(launcher, "find_browser", return_value="/usr/bin/google-chrome"):
                with patch.object(launcher.subprocess, "Popen", return_value=process) as popen:
                    returned, mode = launcher.open_prisma_window(config)

        self.assertIs(returned.process, process)
        self.assertIn("respaldo", mode)
        self.assertIn("no verifica instalación PWA", mode)
        self.assertEqual(
            popen.call_args.args[0],
            [
                "/usr/bin/google-chrome",
                f"--user-data-dir={tmpdir}",
                "--remote-debugging-port=0",
                "--no-first-run",
                "--disable-background-mode",
                "--start-fullscreen",
                "--app=http://127.0.0.1:5000",
            ],
        )

    def test_windows_virtual_environment_python_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python = root / ".venv" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.object(launcher, "WINDOWS", True), patch.object(launcher, "PROJECT_ROOT", root):
                self.assertEqual(launcher.project_python({}), python)

    def test_windows_browser_search_finds_edge_in_program_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            program_files = Path(tmpdir)
            edge = program_files / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            edge.parent.mkdir(parents=True)
            edge.touch()
            environment = {"PROGRAMFILES": str(program_files), "PROGRAMFILES(X86)": "", "LOCALAPPDATA": ""}
            with patch.object(launcher, "WINDOWS", True), patch.dict(launcher.os.environ, environment):
                with patch.object(launcher.shutil, "which", return_value=None):
                    self.assertEqual(launcher.find_browser({"browser_candidates": []}), str(edge))

    def test_windows_default_profile_uses_local_app_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(launcher, "WINDOWS", True), patch.dict(
                launcher.os.environ, {"LOCALAPPDATA": tmpdir}
            ):
                self.assertEqual(
                    launcher.browser_profile_dir({}),
                    Path(tmpdir) / "Prisma" / "browser-profile",
                )

    def test_windows_process_group_flags_are_used_for_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python = root / "python.exe"
            entry = root / "prisma.py"
            python.touch()
            entry.touch()
            process = Mock()
            with patch.object(launcher, "WINDOWS", True), patch.object(launcher, "PROJECT_ROOT", root), patch.object(
                launcher, "LAUNCHER_DIR", root / "launcher"
            ), patch.object(launcher, "ENTRY_POINT", entry), patch.object(
                launcher, "project_python", return_value=python
            ), patch.object(launcher.subprocess, "Popen", return_value=process) as popen:
                returned, log = launcher.start_backend({})
                log.close()

        self.assertIs(returned, process)
        self.assertEqual(popen.call_args.kwargs["creationflags"], 0x00000200)
        self.assertNotIn("start_new_session", popen.call_args.kwargs)

    def test_windows_owned_backend_receives_break_signal(self) -> None:
        process = Mock()
        process.poll.return_value = None
        with patch.object(launcher, "WINDOWS", True):
            launcher.stop_owned_backend(process)
        process.send_signal.assert_called_once_with(getattr(launcher.signal, "CTRL_BREAK_EVENT", launcher.signal.SIGTERM))
        process.terminate.assert_not_called()

    def test_installed_app_id_uses_distinct_chrome_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "base_url": "http://127.0.0.1:5000",
                "installed_app_id": "example-app-id",
                "browser_candidates": ["google-chrome"],
                "browser_profile_dir": tmpdir,
                "fullscreen": False,
            }
            with patch.object(launcher, "find_browser", return_value="/usr/bin/google-chrome"):
                with patch.object(launcher.subprocess, "Popen", return_value=Mock()) as popen:
                    _, mode = launcher.open_prisma_window(config)

        self.assertIn("PWA instalada", mode)
        self.assertNotIn("--start-fullscreen", popen.call_args.args[0])
        self.assertEqual(popen.call_args.args[0][-1], "--app-id=example-app-id")

    def test_window_tracking_does_not_depend_on_open_command_lifetime(self) -> None:
        process = Mock()
        process.poll.return_value = 0
        session = launcher.BrowserSession(process, Path("/unused-profile"), "http://127.0.0.1:5000")
        with patch.object(session, "_prisma_window_is_open", side_effect=[True, True, False]):
            with patch.object(launcher.time, "sleep"):
                session.wait_until_closed()

        self.assertEqual(process.poll.call_count, 0)

    def test_ready_backend_is_reused_without_starting_duplicate(self) -> None:
        config = launcher.load_config()
        browser = Mock()
        with patch.object(launcher, "load_config", return_value=config):
            with patch.object(launcher, "launcher_lock", return_value=nullcontext()):
                with patch.object(launcher, "prisma_is_ready", return_value=True):
                    with patch.object(launcher, "start_backend") as start:
                        with patch.object(
                            launcher,
                            "open_prisma_window",
                            return_value=(browser, "ventana de prueba"),
                        ):
                            self.assertEqual(launcher.run(), 0)
        start.assert_not_called()

    def test_owned_backend_stops_only_after_real_window_closes(self) -> None:
        config = launcher.load_config()
        config["auto_start_backend"] = True
        config["stop_backend_on_exit"] = True
        backend = Mock()
        backend_log = Mock()
        browser = Mock()
        order: list[str] = []
        browser.wait_until_closed.side_effect = lambda: order.append("window_closed")

        with patch.object(launcher, "load_config", return_value=config):
            with patch.object(launcher, "launcher_lock", return_value=nullcontext()):
                with patch.object(launcher, "prisma_is_ready", return_value=False):
                    with patch.object(launcher, "configured_port_is_open", return_value=False):
                        with patch.object(launcher, "start_backend", return_value=(backend, backend_log)):
                            with patch.object(launcher, "wait_for_backend"):
                                with patch.object(
                                    launcher, "open_prisma_window", return_value=(browser, "ventana de prueba")
                                ):
                                    with patch.object(
                                        launcher,
                                        "stop_owned_backend",
                                        side_effect=lambda process: order.append("backend_stopped"),
                                    ) as stop:
                                        self.assertEqual(launcher.run(), 0)

        self.assertEqual(order, ["window_closed", "backend_stopped"])
        stop.assert_called_once_with(backend)

    def test_stop_disabled_leaves_owned_backend_running(self) -> None:
        config = launcher.load_config()
        config["auto_start_backend"] = True
        config["stop_backend_on_exit"] = False
        backend = Mock()
        backend_log = Mock()
        browser = Mock()

        with patch.object(launcher, "load_config", return_value=config):
            with patch.object(launcher, "launcher_lock", return_value=nullcontext()):
                with patch.object(launcher, "prisma_is_ready", return_value=False):
                    with patch.object(launcher, "configured_port_is_open", return_value=False):
                        with patch.object(launcher, "start_backend", return_value=(backend, backend_log)):
                            with patch.object(launcher, "wait_for_backend"):
                                with patch.object(
                                    launcher, "open_prisma_window", return_value=(browser, "ventana de prueba")
                                ):
                                    with patch.object(launcher, "stop_owned_backend") as stop:
                                        self.assertEqual(launcher.run(), 0)

        browser.wait_until_closed.assert_not_called()
        stop.assert_not_called()

    def test_existing_backend_is_never_stopped_by_launcher(self) -> None:
        config = launcher.load_config()
        config["stop_backend_on_exit"] = True
        browser = Mock()
        with patch.object(launcher, "load_config", return_value=config):
            with patch.object(launcher, "launcher_lock", return_value=nullcontext()):
                with patch.object(launcher, "prisma_is_ready", return_value=True):
                    with patch.object(
                        launcher, "open_prisma_window", return_value=(browser, "ventana de prueba")
                    ):
                        with patch.object(launcher, "stop_owned_backend") as stop:
                            self.assertEqual(launcher.run(), 0)

        browser.wait_until_closed.assert_not_called()
        stop.assert_not_called()

    def test_disabled_auto_start_opens_without_starting_python(self) -> None:
        config = launcher.load_config()
        config["auto_start_backend"] = False
        browser = Mock()
        with patch.object(launcher, "load_config", return_value=config):
            with patch.object(launcher, "launcher_lock", return_value=nullcontext()):
                with patch.object(launcher, "prisma_is_ready", return_value=False):
                    with patch.object(launcher, "configured_port_is_open", return_value=False):
                        with patch.object(launcher, "start_backend") as start:
                            with patch.object(
                                launcher,
                                "open_prisma_window",
                                return_value=(browser, "ventana de prueba"),
                            ):
                                self.assertEqual(launcher.run(), 0)
        start.assert_not_called()

    def test_wrong_service_on_port_is_not_opened(self) -> None:
        config = launcher.load_config()
        with patch.object(launcher, "load_config", return_value=config):
            with patch.object(launcher, "launcher_lock", return_value=nullcontext()):
                with patch.object(launcher, "prisma_is_ready", return_value=False):
                    with patch.object(launcher, "configured_port_is_open", return_value=True):
                        with patch.object(launcher, "open_prisma_window") as open_window:
                            self.assertEqual(launcher.run(), 1)
        open_window.assert_not_called()


if __name__ == "__main__":
    unittest.main()
