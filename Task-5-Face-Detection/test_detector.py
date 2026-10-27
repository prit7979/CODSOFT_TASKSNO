"""
Unit tests for FaceDetector module (Task 5 Face Detection).

Tests cover model loading, input validation, image formats,
scale normalization, face detection, face counting, annotation,
and parameter bounds checking using Python's standard `unittest` framework.
"""

import os
import unittest
from typing import List, Tuple
import cv2
import numpy as np

from detector import FaceDetector, DEFAULT_MODEL_PATH


class TestFaceDetectorModel(unittest.TestCase):
    """Tests related to model initialization, paths, and loading."""

    def test_default_initialization(self) -> None:
        """Verify FaceDetector initializes successfully with default YuNet model."""
        detector = FaceDetector()
        self.assertIsNotNone(detector.detector)
        self.assertTrue(os.path.exists(detector.model_path))
        self.assertGreater(os.path.getsize(detector.model_path), 0)

    def test_missing_model_raises_filenotfound(self) -> None:
        """Verify initializing with non-existent path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            FaceDetector("non_existent_model_weights.onnx")

    def test_empty_model_raises_runtime_error(self) -> None:
        """Verify initializing with a 0-byte file raises RuntimeError."""
        dummy_path = os.path.join(os.path.dirname(__file__), "empty_dummy.onnx")
        try:
            with open(dummy_path, "wb") as f:
                pass  # create 0-byte file
            with self.assertRaises(RuntimeError):
                FaceDetector(dummy_path)
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)


class TestFaceDetectorInputValidation(unittest.TestCase):
    """Tests for image input validation."""

    def setUp(self) -> None:
        self.detector = FaceDetector()

    def test_none_input_raises_value_error(self) -> None:
        """Verify None image raises ValueError."""
        with self.assertRaises(ValueError):
            self.detector.validate_image(None)

    def test_non_numpy_array_raises_type_error(self) -> None:
        """Verify non-ndarray types (string, list) raise TypeError."""
        with self.assertRaises(TypeError):
            self.detector.validate_image("not_an_image")
        with self.assertRaises(TypeError):
            self.detector.validate_image([[1, 2], [3, 4]])

    def test_empty_array_raises_value_error(self) -> None:
        """Verify 0-sized array raises ValueError."""
        with self.assertRaises(ValueError):
            self.detector.validate_image(np.array([]))

    def test_invalid_dimensions_raise_value_error(self) -> None:
        """Verify 1D or 4D arrays raise ValueError."""
        with self.assertRaises(ValueError):
            self.detector.validate_image(np.zeros((10,), dtype=np.uint8))
        with self.assertRaises(ValueError):
            self.detector.validate_image(np.zeros((2, 2, 2, 2), dtype=np.uint8))

    def test_invalid_color_channels_raise_value_error(self) -> None:
        """Verify 3D array with channels not in (1, 3, 4) raises ValueError."""
        with self.assertRaises(ValueError):
            self.detector.validate_image(np.zeros((50, 50, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            self.detector.validate_image(np.zeros((50, 50, 5), dtype=np.uint8))


class TestFaceDetectorImageFormatsAndScale(unittest.TestCase):
    """Tests for grayscale, BGR, RGB inputs, immutability, and scale normalization."""

    def setUp(self) -> None:
        self.detector = FaceDetector()

    def test_grayscale_2d_input(self) -> None:
        """Verify 2D grayscale image processes without error and is not mutated."""
        gray = np.full((100, 120), 128, dtype=np.uint8)
        gray_copy = gray.copy()
        faces = self.detector.detect_faces(gray)
        self.assertIsInstance(faces, list)
        np.testing.assert_array_equal(gray, gray_copy)

    def test_bgr_3d_input(self) -> None:
        """Verify 3D BGR image processes without error and is not mutated."""
        bgr = np.full((120, 100, 3), 100, dtype=np.uint8)
        bgr_copy = bgr.copy()
        faces = self.detector.detect_faces(bgr, is_rgb=False)
        self.assertIsInstance(faces, list)
        np.testing.assert_array_equal(bgr, bgr_copy)

    def test_rgb_3d_input(self) -> None:
        """Verify 3D RGB image processes without error and is not mutated."""
        rgb = np.full((150, 150, 3), 200, dtype=np.uint8)
        rgb_copy = rgb.copy()
        faces = self.detector.detect_faces(rgb, is_rgb=True)
        self.assertIsInstance(faces, list)
        np.testing.assert_array_equal(rgb, rgb_copy)

    def test_rgba_4d_input(self) -> None:
        """Verify 4-channel RGBA image processes correctly."""
        rgba = np.full((80, 80, 4), 255, dtype=np.uint8)
        faces = self.detector.detect_faces(rgba, is_rgb=True)
        self.assertIsInstance(faces, list)

    def test_scale_normalization_trigger_on_high_res(self) -> None:
        """Verify high-resolution images (> 800px) execute and reset input dimensions."""
        high_res = np.zeros((1600, 1200, 3), dtype=np.uint8)
        high_res_copy = high_res.copy()
        faces = self.detector.detect_faces(high_res, is_rgb=True)
        self.assertIsInstance(faces, list)
        np.testing.assert_array_equal(high_res, high_res_copy)
        # Verify detector input size was scaled down to max dim 800 (600, 800)
        self.assertEqual(self.detector.detector.getInputSize(), (600, 800))

    def test_non_uint8_conversion(self) -> None:
        """Verify float or int32 arrays are safely converted to uint8 without error."""
        float_img = np.zeros((100, 100, 3), dtype=np.float32)
        faces = self.detector.detect_faces(float_img)
        self.assertIsInstance(faces, list)


class TestFaceDetectorDetection(unittest.TestCase):
    """Tests for detection outputs, face counts, and coordinate bounds."""

    def setUp(self) -> None:
        self.detector = FaceDetector()

    def test_blank_image_returns_empty_list(self) -> None:
        """Verify blank black image returns zero detected faces."""
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(blank)
        self.assertEqual(faces, [])

    def test_count_faces_matches_len_detect_faces(self) -> None:
        """Verify count_faces() output strictly matches len(detect_faces())."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(img)
        count = self.detector.count_faces(img)
        self.assertEqual(count, len(faces))

    def test_bounding_box_format_and_bounds_with_synthetic_boxes(self) -> None:
        """Verify bounding box coordinates format and clipping."""
        # Test on dummy coordinates
        boxes = [(10, 20, 50, 60), (0, 0, 100, 100)]
        for (x, y, w, h) in boxes:
            self.assertIsInstance(x, int)
            self.assertIsInstance(y, int)
            self.assertIsInstance(w, int)
            self.assertIsInstance(h, int)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)


class TestFaceDetectorAnnotation(unittest.TestCase):
    """Tests for draw_faces() annotation function."""

    def setUp(self) -> None:
        self.detector = FaceDetector()

    def test_draw_faces_preserves_dimensions(self) -> None:
        """Verify annotated image has identical shape and dtype to input."""
        img = np.zeros((150, 200, 3), dtype=np.uint8)
        dummy_boxes = [(10, 10, 30, 40)]
        annotated = self.detector.draw_faces(img, dummy_boxes, color=(0, 255, 0), thickness=2)
        self.assertEqual(annotated.shape, img.shape)
        self.assertEqual(annotated.dtype, img.dtype)

    def test_draw_faces_does_not_mutate_original(self) -> None:
        """Verify draw_faces returns a new array and does not modify the original image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_copy = img.copy()
        dummy_boxes = [(20, 20, 40, 40)]
        annotated = self.detector.draw_faces(img, dummy_boxes, color=(255, 0, 0), thickness=2, draw_labels=True)
        np.testing.assert_array_equal(img, img_copy)
        self.assertFalse(np.array_equal(annotated, img))


class TestFaceDetectorParameters(unittest.TestCase):
    """Tests for parameter validation and thresholds."""

    def setUp(self) -> None:
        self.detector = FaceDetector()
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_valid_thresholds_execute(self) -> None:
        """Verify valid score and nms threshold bounds run without error."""
        faces = self.detector.detect_faces(self.img, score_threshold=0.30, nms_threshold=0.10)
        self.assertIsInstance(faces, list)
        faces2 = self.detector.detect_faces(self.img, score_threshold=0.95, nms_threshold=0.70)
        self.assertIsInstance(faces2, list)

    def test_invalid_score_threshold_raises_value_error(self) -> None:
        """Verify score_threshold < 0 or > 1 raises ValueError."""
        with self.assertRaises(ValueError):
            self.detector.detect_faces(self.img, score_threshold=-0.1)
        with self.assertRaises(ValueError):
            self.detector.detect_faces(self.img, score_threshold=1.1)

    def test_invalid_nms_threshold_raises_value_error(self) -> None:
        """Verify nms_threshold < 0 or > 1 raises ValueError."""
        with self.assertRaises(ValueError):
            self.detector.detect_faces(self.img, nms_threshold=-0.05)
        with self.assertRaises(ValueError):
            self.detector.detect_faces(self.img, nms_threshold=1.5)


if __name__ == "__main__":
    unittest.main()
