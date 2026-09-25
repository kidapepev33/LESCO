from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.web_api import DEFAULT_RESULT_PATH, create_app, read_current_result  # noqa: E402


class WebApiTests(unittest.TestCase):
    def test_default_result_path_stays_at_project_root(self) -> None:
        self.assertEqual(DEFAULT_RESULT_PATH, ROOT / "godot_bridge" / "output.txt")

    def test_reads_real_godot_output_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.txt"
            output.write_text(
                "Oración: HOLA\nScore visual: 0.910\nDetecciones:\nHOLA conf=0.910",
                encoding="utf-8",
            )
            self.assertEqual(read_current_result(output), {"seña": "HOLA", "confianza": 0.91})

    def test_empty_or_missing_result_is_waiting(self) -> None:
        missing = Path("/tmp/lesco-result-that-does-not-exist.txt")
        self.assertEqual(read_current_result(missing), {"seña": "", "confianza": None})

    def test_resultado_endpoint_returns_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.txt"
            output.write_text("Oración: DORMIR\nScore visual: 0.998", encoding="utf-8")
            client = create_app(output).test_client()
            response = client.get("/resultado")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"seña": "DORMIR", "confianza": 0.998})
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_root_serves_test_page(self) -> None:
        response = create_app().test_client().get("/")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Prisma", response.data)
            self.assertIn(b'/assets/js/app.js', response.data)
        finally:
            response.close()

    def test_web_assets_are_served(self) -> None:
        client = create_app().test_client()
        stylesheet = client.get("/assets/css/main.css")
        script = client.get("/assets/js/app.js")
        try:
            self.assertEqual(stylesheet.status_code, 200)
            self.assertEqual(script.status_code, 200)
            self.assertIn(b"fetch('/resultado'", script.data)
        finally:
            stylesheet.close()
            script.close()

    def test_all_independent_pages_are_served(self) -> None:
        client = create_app().test_client()
        for route in ("/", "/lesco-a-texto", "/texto-a-lesco", "/nosotros", "/ayuda"):
            response = client.get(route)
            try:
                self.assertEqual(response.status_code, 200, route)
                self.assertIn(b'/assets/css/main.css', response.data)
            finally:
                response.close()

    def test_frame_serves_existing_recognizer_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = Path(tmpdir) / "frame.jpg"
            frame.write_bytes(b"existing-camera-frame")
            response = create_app(frame_path=frame).test_client().get("/frame")
            try:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, b"existing-camera-frame")
                self.assertEqual(response.mimetype, "image/jpeg")
                self.assertIn("no-store", response.headers["Cache-Control"])
            finally:
                response.close()


if __name__ == "__main__":
    unittest.main()
