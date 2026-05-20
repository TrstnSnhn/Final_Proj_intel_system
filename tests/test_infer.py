import unittest

from src.infer import format_topk_predictions


class InferenceFormattingTests(unittest.TestCase):
    def test_formats_top_k_predictions_in_confidence_order(self):
        predictions = format_topk_predictions(
            class_names=["Apple___healthy", "Tomato___Early_blight", "Squash___Powdery_mildew"],
            probabilities=[0.15, 0.7, 0.15],
            top_k=2,
        )

        self.assertEqual(
            predictions,
            [
                {"rank": 1, "class_name": "Tomato___Early_blight", "confidence": 0.7},
                {"rank": 2, "class_name": "Apple___healthy", "confidence": 0.15},
            ],
        )

    def test_rejects_top_k_larger_than_class_count(self):
        with self.assertRaisesRegex(ValueError, "top_k cannot exceed"):
            format_topk_predictions(["Apple___healthy"], [1.0], top_k=2)


if __name__ == "__main__":
    unittest.main()
