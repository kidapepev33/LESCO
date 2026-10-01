from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prisma  # noqa: E402


class PrismaLauncherTests(unittest.TestCase):
    def test_declares_only_runtime_components(self) -> None:
        scripts = [component.script.relative_to(ROOT).as_posix() for component in prisma.COMPONENTS]
        self.assertEqual(
            scripts,
            ["web/web_api.py", "src/sign_video_bridge.py", "src/predict_live.py"],
        )
        self.assertNotIn("train_model.py", " ".join(scripts))
        self.assertNotIn("record_sign.py", " ".join(scripts))

    def test_start_uses_current_python_and_disables_shared_process_groups(self) -> None:
        children = [Mock(pid=101), Mock(pid=102), Mock(pid=103)]
        popen = Mock(side_effect=children)

        started = prisma.start_components(popen_factory=popen)

        self.assertEqual([child for _, child in started], children)
        self.assertEqual(popen.call_count, 3)
        commands = [call.args[0] for call in popen.call_args_list]
        self.assertTrue(all(command[:2] == [sys.executable, "-u"] for command in commands))
        self.assertEqual(
            [Path(command[2]).relative_to(ROOT).as_posix() for command in commands],
            ["web/web_api.py", "src/sign_video_bridge.py", "src/predict_live.py"],
        )
        self.assertTrue(all(call.kwargs["cwd"] == ROOT for call in popen.call_args_list))
        self.assertTrue(all(call.kwargs["start_new_session"] for call in popen.call_args_list))

    def test_windows_components_use_new_process_groups(self) -> None:
        children = [Mock(pid=101), Mock(pid=102), Mock(pid=103)]
        with patch.object(prisma, "WINDOWS", True):
            popen = Mock(side_effect=children)
            prisma.start_components(popen_factory=popen)
        self.assertTrue(all(call.kwargs["creationflags"] == 0x00000200 for call in popen.call_args_list))
        self.assertTrue(all("start_new_session" not in call.kwargs for call in popen.call_args_list))

    def test_windows_graceful_stop_sends_break_event(self) -> None:
        process = Mock(pid=321)
        process.poll.return_value = None
        with patch.object(prisma, "WINDOWS", True):
            prisma.request_graceful_stop(process)
        process.send_signal.assert_called_once_with(getattr(prisma.signal, "CTRL_BREAK_EVENT", prisma.signal.SIGTERM))

    @unittest.skipIf(prisma.os.name == "nt", "La señal de grupo se prueba en sistemas POSIX.")
    def test_graceful_stop_sends_sigint_to_child_group(self) -> None:
        process = Mock(pid=321)
        process.poll.return_value = None

        with patch.object(prisma.os, "killpg") as killpg:
            prisma.request_graceful_stop(process)

        killpg.assert_called_once_with(321, prisma.signal.SIGINT)

    def test_stop_escalates_when_child_ignores_sigint(self) -> None:
        component = prisma.COMPONENTS[0]
        process = Mock(pid=123)
        process.poll.side_effect = [None, None, 0]
        process.wait.side_effect = [subprocess.TimeoutExpired("web", 5), None]

        with patch.object(prisma, "request_graceful_stop"):
            prisma.stop_components([(component, process)])

        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()

    def test_partial_start_failure_closes_process_already_started(self) -> None:
        first_child = Mock(pid=101)
        popen = Mock(side_effect=[first_child, OSError("falló el segundo proceso")])

        with patch.object(prisma, "stop_components") as stop:
            with self.assertRaises(OSError):
                prisma.start_components(popen_factory=popen)

        stop.assert_called_once()
        self.assertEqual(stop.call_args.args[0][0][1], first_child)


if __name__ == "__main__":
    unittest.main()
