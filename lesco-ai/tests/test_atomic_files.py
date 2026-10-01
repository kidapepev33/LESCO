from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.utils.atomic_files import replace_with_retry, write_text_atomic


class AtomicFileTests(unittest.TestCase):
    def test_replace_retries_temporary_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            destination = Path(tmpdir) / "destination"
            source.write_text("new", encoding="utf-8")
            real_replace = __import__("os").replace
            calls = 0

            def sometimes_locked(left, right):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError("sharing violation")
                real_replace(left, right)

            with patch("src.utils.atomic_files.os.replace", side_effect=sometimes_locked), patch(
                "src.utils.atomic_files.time.sleep"
            ) as sleep:
                replace_with_retry(source, destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), "new")
            self.assertEqual(sleep.call_count, 2)

    def test_atomic_text_keeps_content_and_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "debug_response.txt"
            write_text_atomic(destination, "same format\n")
            self.assertEqual(destination.read_text(encoding="utf-8"), "same format\n")
            self.assertFalse((Path(tmpdir) / ".debug_response.txt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
