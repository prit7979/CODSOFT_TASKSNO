"""
Unit Tests for AI Image Captioning Engine
=========================================
CodSoft Artificial Intelligence Internship - Task 3
Verifies multimodal caption generation, native OCR, color/composition analysis,
and error handling using Python's built-in unittest framework and mocks.
"""

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure Task-3 directory is on path regardless of invocation CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image, ImageDraw, ImageFont

from image_analysis import analyze_composition, analyze_image_colors, rgb_to_color_name
from model import (
    CaptionGenerationError,
    CaptionResult,
    ModelLoadError,
    generate_caption,
    load_captioning_model,
)
from ocr_engine import clean_extracted_text, extract_visible_text


class TestImageCaptioningEngine(unittest.TestCase):
    """Test suite covering input validation, OCR, image preprocessing, and inference wrappers."""

    def setUp(self):
        """Prepare sample RGB and non-RGB test images in memory."""
        self.rgb_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        self.rgba_image = Image.new("RGBA", (100, 100), color=(0, 255, 0, 128))
        self.grayscale_image = Image.new("L", (100, 100), color=128)

    def test_invalid_input_type_raises_caption_error(self):
        """Passing a non-Image object must raise CaptionGenerationError."""
        fake_processor = MagicMock()
        fake_model = MagicMock()

        for invalid_input in ["not_an_image", None, 12345, [1, 2, 3]]:
            with self.assertRaises(CaptionGenerationError):
                generate_caption(fake_processor, fake_model, invalid_input)

    def test_color_naming_accuracy(self):
        """Verify RGB colors map to appropriate descriptive names."""
        self.assertEqual(rgb_to_color_name(255, 228, 235), "light pink")
        self.assertEqual(rgb_to_color_name(64, 224, 208), "turquoise / teal")
        self.assertEqual(rgb_to_color_name(255, 255, 255), "pure white")
        self.assertEqual(rgb_to_color_name(34, 139, 34), "vibrant green")
        self.assertEqual(rgb_to_color_name(139, 90, 43), "wooden brown")

    def test_ocr_clean_extracted_text(self):
        """Verify noise stripping and deduplication in OCR text cleaner."""
        raw_lines = ["", "!@#$", "  SupriLicious Cakes  ", "supriLicious cakes", "Custom Cakes"]
        cleaned = clean_extracted_text(raw_lines)
        self.assertEqual(cleaned, ["SupriLicious Cakes", "Custom Cakes"])

    def test_ocr_on_text_containing_image(self):
        """Verify native OCR extracts text rendered on a synthetic canvas."""
        img = Image.new("RGB", (600, 150), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except Exception:
            font = ImageFont.load_default()

        draw.text((30, 40), "Coffee & Bakery", fill=(0, 0, 0), font=font)
        res = extract_visible_text(img)

        self.assertTrue(res["has_text"])
        self.assertTrue(any("Coffee" in line for line in res["lines"]))

    def test_ocr_on_non_text_image(self):
        """Verify native OCR reports has_text=False on a plain color block."""
        plain_img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        res = extract_visible_text(plain_img)
        self.assertFalse(res["has_text"])
        self.assertEqual(res["lines"], [])

    def test_composition_landscape_and_portrait(self):
        """Verify aspect ratio classification in composition analysis."""
        wide_img = Image.new("RGB", (800, 400), color=(200, 200, 200))
        comp_wide = analyze_composition(wide_img)
        self.assertEqual(comp_wide["aspect_type"], "landscape")

        tall_img = Image.new("RGB", (300, 600), color=(200, 200, 200))
        comp_tall = analyze_composition(tall_img)
        self.assertEqual(comp_tall["aspect_type"], "portrait")

    def test_caption_modes_synthesis(self):
        """Verify short, detailed, and very_detailed caption generation with mock model."""
        fake_processor = MagicMock()
        fake_model = MagicMock()

        fake_processor.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}
        fake_processor.decode.side_effect = [
            "a cupcake with frosting",
            "the background features balloons and bunting",
        ]
        fake_model.generate.return_value = torch.tensor([[101, 102]])

        # Test Short Mode
        short_res = generate_caption(fake_processor, fake_model, self.rgb_image, mode="short")
        self.assertIsInstance(short_res, CaptionResult)
        self.assertEqual(short_res.mode, "short")
        self.assertIn("cupcake", str(short_res).lower())

        # Reset decode side effect
        fake_processor.decode.side_effect = [
            "a cupcake with frosting",
            "the background features balloons and bunting",
        ]
        # Test Detailed Mode
        det_res = generate_caption(fake_processor, fake_model, self.rgb_image, mode="detailed")
        self.assertEqual(det_res.mode, "detailed")
        self.assertIn("cupcake", str(det_res).lower())

        # Reset decode side effect
        fake_processor.decode.side_effect = [
            "a cupcake with frosting",
            "the background features balloons and bunting",
        ]
        # Test Very Detailed Mode
        vdet_res = generate_caption(fake_processor, fake_model, self.rgb_image, mode="very_detailed")
        self.assertEqual(vdet_res.mode, "very_detailed")
        self.assertIn("Primary Subject", str(vdet_res))
        self.assertIn("Color Palette", str(vdet_res))

    def test_model_inference_exception_wrapped_cleanly(self):
        """If model.generate raises a runtime error, it must be caught as CaptionGenerationError."""
        fake_processor = MagicMock()
        fake_model = MagicMock()
        fake_processor.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}
        fake_model.generate.side_effect = RuntimeError("Simulated CPU memory failure")

        with self.assertRaises(CaptionGenerationError) as ctx:
            generate_caption(fake_processor, fake_model, self.rgb_image)

        self.assertIn("Failed during caption generation", str(ctx.exception))

    def test_benchmark_samples_exist_and_valid(self):
        """Verify all benchmark images exist and can be loaded with Pillow."""
        benchmarks_dir = os.path.join(os.path.dirname(__file__), "benchmark_samples")
        expected_files = ["bakery_logo.png", "landscape_sample.jpg", "animal_sample.jpg", "person_sample.jpg"]

        for fname in expected_files:
            fpath = os.path.join(benchmarks_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing benchmark sample: {fname}")
            with Image.open(fpath) as img:
                self.assertGreater(img.size[0], 0)
                self.assertGreater(img.size[1], 0)


if __name__ == "__main__":
    unittest.main()
