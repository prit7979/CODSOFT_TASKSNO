"""
Face Detection Module using OpenCV YuNet Deep Learning Model.

This module provides a beginner-friendly, type-hinted `FaceDetector` class
powered by OpenCV's built-in `cv2.FaceDetectorYN` and the pre-trained YuNet ONNX
neural network model. It performs fast, accurate face localization on CPU with
configurable confidence and Non-Maximum Suppression (NMS) thresholds.
"""

import os
from typing import List, Tuple, Optional, Any
import cv2
import numpy as np

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    """
    Face detector utilizing OpenCV's deep learning YuNet ONNX model.

    Attributes:
        model_path (str): Filepath to the YuNet ONNX model file.
        detector (cv2.FaceDetectorYN): Initialized OpenCV YuNet detector instance.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initializes the FaceDetector and loads the YuNet ONNX model on CPU.

        Args:
            model_path: Optional custom path to the YuNet ONNX model file.
                If omitted or None, defaults to 'models/face_detection_yunet_2023mar.onnx'
                relative to this file.

        Raises:
            FileNotFoundError: If the ONNX model file cannot be found.
            RuntimeError: If OpenCV fails to initialize the FaceDetectorYN model.
        """
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet ONNX model file not found at: '{model_path}'."
            )

        if os.path.getsize(model_path) == 0:
            raise RuntimeError(
                f"YuNet ONNX model file is empty (0 bytes): '{model_path}'."
            )

        self.model_path: str = os.path.abspath(model_path)

        try:
            self.detector: cv2.FaceDetectorYN = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.7,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize cv2.FaceDetectorYN with model '{self.model_path}': {exc}"
            ) from exc

        if self.detector is None:
            raise RuntimeError(
                f"OpenCV failed to instantiate FaceDetectorYN from: '{self.model_path}'."
            )

    @staticmethod
    def validate_image(image: Any) -> None:
        """
        Validates the input image structure, type, and dimensions.

        Args:
            image: Image object to validate.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the array is empty or lacks valid image dimensions (2D or 3D).
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Input image must be a numpy.ndarray, got {type(image).__name__}."
            )

        if image.size == 0:
            raise ValueError("Input image is empty (contains zero pixels).")

        if image.ndim not in (2, 3):
            raise ValueError(
                f"Input image must have 2 (grayscale) or 3 (color) dimensions, "
                f"got {image.ndim} dimensions with shape {image.shape}."
            )

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Expected 1, 3, or 4 color channels, but image has {image.shape[2]} channels."
            )

    def _prepare_bgr_image(self, image: np.ndarray, is_rgb: bool = False) -> np.ndarray:
        """
        Ensures the input image is formatted as a 3-channel BGR uint8 array
        expected by the YuNet convolutional neural network.

        Args:
            image: Input image array (2D grayscale, 3-channel BGR/RGB, or 4-channel BGRA/RGBA).
            is_rgb: True if color channels are in RGB/RGBA order (e.g. from PIL/Streamlit).

        Returns:
            np.ndarray: 3-channel BGR uint8 image.
        """
        # Ensure uint8 data type
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # 2D grayscale -> 3-channel BGR
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Single channel with dimension (H, W, 1) -> 3-channel BGR
        if image.shape[2] == 1:
            return cv2.cvtColor(image.squeeze(axis=2), cv2.COLOR_GRAY2BGR)

        # 3-channel RGB -> BGR
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image

        # 4-channel RGBA / BGRA -> 3-channel BGR
        if image.shape[2] == 4:
            conversion = cv2.COLOR_RGBA2BGR if is_rgb else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(image, conversion)

        return image

    def detect_faces(
        self,
        image: np.ndarray,
        score_threshold: float = 0.70,
        nms_threshold: float = 0.30,
        is_rgb: bool = False,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in an image using the pre-trained YuNet deep learning model.

        Args:
            image: Input image as a numpy.ndarray.
            score_threshold: Confidence threshold (0.0 to 1.0) to filter out weak detections.
                Default 0.70. Higher values reduce false positives.
            nms_threshold: Non-Maximum Suppression IoU threshold (0.0 to 1.0) to suppress
                overlapping bounding boxes. Default 0.30.
            is_rgb: Set to True if the input image is in RGB color space (e.g. from Streamlit).

        Returns:
            List[Tuple[int, int, int, int]]: A list of detected bounding boxes, where each box
            is represented as (x, y, w, h) in pixels. The count of detected faces is
            simply `len(faces)`.

        Raises:
            ValueError: If score_threshold or nms_threshold is out of valid [0.0, 1.0] bounds.
        """
        if not (0.0 <= score_threshold <= 1.0):
            raise ValueError(
                f"score_threshold must be between 0.0 and 1.0, got {score_threshold}."
            )

        if not (0.0 <= nms_threshold <= 1.0):
            raise ValueError(
                f"nms_threshold must be between 0.0 and 1.0, got {nms_threshold}."
            )

        self.validate_image(image)
        bgr = self._prepare_bgr_image(image, is_rgb=is_rgb)

        orig_h, orig_w = bgr.shape[:2]
        max_dim = max(orig_h, orig_w)

        # Scale normalization: if the longest dimension exceeds 800 pixels,
        # proportionally resize the image to fit YuNet's optimal receptive field.
        # This prevents confidence score collapse on high-resolution close-up portraits.
        if max_dim > 800:
            scale = 800.0 / float(max_dim)
            infer_w = max(1, int(round(orig_w * scale)))
            infer_h = max(1, int(round(orig_h * scale)))
            infer_bgr = cv2.resize(
                bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA
            )
        else:
            scale = 1.0
            infer_w, infer_h = orig_w, orig_h
            infer_bgr = bgr

        # Dynamically adapt detector input dimensions to match current inference shape (W, H)
        self.detector.setInputSize((int(infer_w), int(infer_h)))
        self.detector.setScoreThreshold(float(score_threshold))
        self.detector.setNMSThreshold(float(nms_threshold))

        _, raw_faces = self.detector.detect(infer_bgr)

        if raw_faces is None or len(raw_faces) == 0:
            return []

        # YuNet output layout per row [N, 15]:
        # indices 0-1: x, y (top-left corner)
        # indices 2-3: width, height
        # indices 4-13: 5 facial landmarks (eyes, nose, mouth corners)
        # index 14: confidence score
        boxes: List[Tuple[int, int, int, int]] = []
        for face in raw_faces:
            x, y, w, h = face[0:4]

            # Invert scale to map bounding box back to original full-resolution coordinate system
            if scale != 1.0:
                x = x / scale
                y = y / scale
                w = w / scale
                h = h / scale

            # Ensure valid integer bounds strictly within original image limits
            x_int = max(0, min(orig_w - 1, int(round(x))))
            y_int = max(0, min(orig_h - 1, int(round(y))))
            w_int = max(1, min(orig_w - x_int, int(round(w))))
            h_int = max(1, min(orig_h - y_int, int(round(h))))

            boxes.append((x_int, y_int, w_int, h_int))

        return boxes

    def count_faces(
        self,
        image: np.ndarray,
        score_threshold: float = 0.70,
        nms_threshold: float = 0.30,
        is_rgb: bool = False,
    ) -> int:
        """
        Convenience method to return the count of detected faces in an image.

        Args:
            image: Input image as a numpy.ndarray.
            score_threshold: Confidence threshold for face detection.
            nms_threshold: Non-Maximum Suppression threshold.
            is_rgb: Whether input is in RGB format.

        Returns:
            int: The total number of detected faces.
        """
        return len(
            self.detect_faces(
                image=image,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                is_rgb=is_rgb,
            )
        )

    def draw_faces(
        self,
        image: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_labels: bool = False,
    ) -> np.ndarray:
        """
        Draws bounding box rectangles around detected faces on a copy of the image.

        Args:
            image: Original input image as a numpy.ndarray.
            faces: List of bounding boxes as (x, y, w, h) tuples.
            color: Rectangle color in RGB/BGR format, default green (0, 255, 0).
            thickness: Line thickness in pixels.
            draw_labels: If True, draws a face index tag (e.g. 'Face #1') above each box.

        Returns:
            np.ndarray: A new annotated image array with drawn bounding boxes.
        """
        self.validate_image(image)
        annotated = image.copy()

        for idx, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            if draw_labels:
                label = f"Face #{idx}"
                label_y = y - 10 if y - 10 > 10 else y + 20
                cv2.putText(
                    annotated,
                    label,
                    (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness=max(1, thickness - 1),
                    lineType=cv2.LINE_AA,
                )

        return annotated
