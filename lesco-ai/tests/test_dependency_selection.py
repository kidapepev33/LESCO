from importlib.metadata import requires
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencySelectionTests(unittest.TestCase):
    def test_only_one_opencv_distribution_is_declared(self) -> None:
        requirements = {
            line.strip().split("==", 1)[0].lower()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("opencv-contrib-python", requirements)
        self.assertNotIn("opencv-python", requirements)

    def test_installed_mediapipe_requires_contrib_opencv(self) -> None:
        dependencies = requires("mediapipe") or []
        self.assertTrue(any(item.lower().startswith("opencv-contrib-python") for item in dependencies))


if __name__ == "__main__":
    unittest.main()
