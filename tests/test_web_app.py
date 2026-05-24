import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from web.app import create_app


def make_image_upload(filename: str = "leaf.jpg") -> tuple[io.BytesIO, str]:
    image = Image.new("RGB", (16, 16), color=(40, 120, 70))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer, filename


class WebAppTests(unittest.TestCase):
    def make_app(self, **config):
        app = create_app(
            {
                "TESTING": True,
                "WTF_CSRF_ENABLED": False,
                **config,
            }
        )
        return app

    def test_get_index_returns_upload_page(self):
        client = self.make_app().test_client()

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PlantGuard", response.data)
        self.assertIn(b"Baseline educational screening", response.data)
        self.assertIn(b"not a diagnosis", response.data)
        self.assertIn(b"jpg, jpeg, png, bmp, webp", response.data)
        self.assertIn(b"Maximum size: 5 MB", response.data)
        self.assertIn(b"/static/styles.css", response.data)
        self.assertIn(b"leaf-preview", response.data)

    def test_stylesheet_route_returns_css(self):
        client = self.make_app().test_client()

        response = client.get("/static/styles.css")
        body = response.get_data()
        response.close()

        self.assertEqual(response.status_code, 200)
        self.assertIn(b".prediction-card", body)
        self.assertIn(b".upload-panel", body)

    def test_healthz_returns_non_sensitive_status_without_model_artifacts(self):
        client = self.make_app(
            PLANTGUARD_CHECKPOINT_PATH="C:/private/model.pt",
            PLANTGUARD_CLASS_MAP_PATH="C:/private/classes.json",
        ).test_client()

        response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"service": "plantguard", "status": "ok"})
        body = response.get_data(as_text=True)
        self.assertNotIn("C:/private", body)
        self.assertNotIn("model.pt", body)
        self.assertNotIn("classes.json", body)

    def test_vercel_wrapper_exports_flask_app(self):
        from flask import Flask

        from app.app import app

        self.assertIsInstance(app, Flask)

    def test_predict_rejects_missing_file(self):
        client = self.make_app().test_client()

        response = client.post("/predict", data={}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Please choose an image file", response.data)

    def test_predict_rejects_invalid_extension(self):
        client = self.make_app().test_client()

        response = client.post(
            "/predict",
            data={"image": (io.BytesIO(b"not an image"), "leaf.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Unsupported file type", response.data)

    def test_predict_rejects_unreadable_image_with_valid_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            class_map = root / "classes.json"
            checkpoint.write_bytes(b"placeholder")
            class_map.write_text('{"classes": ["Tomato_healthy"]}', encoding="utf-8")
            client = self.make_app(
                PLANTGUARD_CHECKPOINT_PATH=str(checkpoint),
                PLANTGUARD_CLASS_MAP_PATH=str(class_map),
            ).test_client()

            response = client.post(
                "/predict",
                data={"image": (io.BytesIO(b"not an image"), "leaf.jpg")},
                content_type="multipart/form-data",
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn(b"could not be read as an image", response.data)

    def test_predict_rejects_oversized_upload(self):
        client = self.make_app(MAX_CONTENT_LENGTH=32).test_client()

        response = client.post(
            "/predict",
            data={"image": (io.BytesIO(b"x" * 128), "leaf.jpg")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn(b"Upload is too large", response.data)

    def test_predict_reports_missing_checkpoint_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_map = root / "classes.json"
            class_map.write_text('{"classes": ["Tomato_healthy"]}', encoding="utf-8")
            client = self.make_app(
                PLANTGUARD_CHECKPOINT_PATH=str(root / "missing.pt"),
                PLANTGUARD_CLASS_MAP_PATH=str(class_map),
            ).test_client()

            response = client.post(
                "/predict",
                data={"image": make_image_upload()},
                content_type="multipart/form-data",
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn(b"Model artifact is missing", response.data)
            self.assertIn(b"Recreate the baseline checkpoint", response.data)

    def test_predict_returns_mocked_predictions_without_real_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            class_map = root / "classes.json"
            checkpoint.write_bytes(b"placeholder")
            class_map.write_text('{"classes": ["Tomato_healthy"]}', encoding="utf-8")
            client = self.make_app(
                PLANTGUARD_CHECKPOINT_PATH=str(checkpoint),
                PLANTGUARD_CLASS_MAP_PATH=str(class_map),
            ).test_client()

            with patch(
                "web.app.predict_image",
                return_value=[
                    {"rank": 1, "class_name": "Tomato_healthy", "confidence": 0.93},
                    {"rank": 2, "class_name": "Tomato_Late_blight", "confidence": 0.04},
                    {"rank": 3, "class_name": "Tomato_Leaf_Mold", "confidence": 0.03},
                ],
            ) as prediction:
                response = client.post(
                    "/predict",
                    data={"image": make_image_upload()},
                    content_type="multipart/form-data",
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Tomato_healthy", response.data)
            self.assertIn(b"93.00%", response.data)
            self.assertIn(b"prediction-card", response.data)
            self.assertIn(b"confidence-bar", response.data)
            self.assertIn(b"Rank 1", response.data)
            prediction.assert_called_once()

    def test_api_predict_rejects_missing_file_with_json_error(self):
        client = self.make_app().test_client()

        response = client.post("/api/predict", data={}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content_type, "application/json")
        self.assertIn("Please choose an image file", response.get_json()["error"])

    def test_api_predict_rejects_invalid_extension_with_json_error(self):
        client = self.make_app().test_client()

        response = client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b"not an image"), "leaf.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content_type, "application/json")
        self.assertIn("Unsupported file type", response.get_json()["error"])

    def test_api_predict_reports_missing_artifacts_with_json_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = self.make_app(
                PLANTGUARD_CHECKPOINT_PATH=str(root / "missing.pt"),
                PLANTGUARD_CLASS_MAP_PATH=str(root / "missing_classes.json"),
            ).test_client()

            response = client.post(
                "/api/predict",
                data={"image": make_image_upload()},
                content_type="multipart/form-data",
            )

            self.assertEqual(response.status_code, 400)
            body = response.get_json()
            self.assertIn("Model artifact is missing", body["error"])

    def test_api_predict_returns_mocked_predictions_as_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            class_map = root / "classes.json"
            checkpoint.write_bytes(b"placeholder")
            class_map.write_text('{"classes": ["Tomato_healthy"]}', encoding="utf-8")
            client = self.make_app(
                PLANTGUARD_CHECKPOINT_PATH=str(checkpoint),
                PLANTGUARD_CLASS_MAP_PATH=str(class_map),
            ).test_client()

            with patch(
                "web.app.predict_image",
                return_value=[
                    {"rank": 1, "class_name": "Tomato_healthy", "confidence": 0.93},
                    {"rank": 2, "class_name": "Tomato_Late_blight", "confidence": 0.04},
                    {"rank": 3, "class_name": "Tomato_Leaf_Mold", "confidence": 0.03},
                ],
            ):
                response = client.post(
                    "/api/predict",
                    data={"image": make_image_upload()},
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["predictions"][0]["class_name"], "Tomato_healthy")
        self.assertEqual(body["predictions"][0]["confidence"], 0.93)
        self.assertIn("not a definitive diagnosis", body["disclaimer"])


if __name__ == "__main__":
    unittest.main()
