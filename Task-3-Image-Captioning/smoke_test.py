"""
Smoke test script for Task 3 AI Image Captioning
================================================
Verifies model loading, weight caching to Drive D, and CPU inference.
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw

# Ensure local module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import load_captioning_model, generate_caption


def create_local_test_image(filepath: str) -> Image.Image:
    """Generate a clean synthetic RGB test image locally with Pillow without network calls."""
    # Create a 384x384 canvas
    img = Image.new("RGB", (384, 384), color=(135, 206, 235))  # Sky blue background
    draw = ImageDraw.Draw(img)

    # Draw green ground
    draw.rectangle([0, 250, 384, 384], fill=(34, 139, 34))

    # Draw yellow sun
    draw.ellipse([40, 40, 110, 110], fill=(255, 215, 0))

    # Draw a small house: red base + brown roof
    draw.rectangle([140, 180, 250, 290], fill=(178, 34, 34))
    draw.polygon([(120, 180), (195, 110), (270, 180)], fill=(139, 69, 19))

    # Save to disk
    img.save(filepath, format="JPEG", quality=95)
    return img


def run_smoke_test() -> None:
    print("=" * 65)
    print("  CodSoft Task 3: AI Image Captioning - Smoke Test")
    print("=" * 65)

    cache_dir = os.environ.get("HF_HOME", "Not Set")
    print(f"HF_HOME configured to: {cache_dir}")
    assert "D:\\AI_Model_Cache" in cache_dir or "D:/AI_Model_Cache" in cache_dir, (
        f"HF_HOME must be pointed to Drive D! Current: {cache_dir}"
    )

    test_img_path = os.path.join(os.path.dirname(__file__), "sample_test_image.jpg")
    print(f"Generating local test image at: {test_img_path}")
    test_image = create_local_test_image(test_img_path)

    print("\n[1/3] Loading BLIP processor and model weights onto CPU...")
    start_load = time.perf_counter()
    processor, model = load_captioning_model()
    load_duration = time.perf_counter() - start_load
    print(f"-> Model successfully loaded in {load_duration:.2f} seconds.")

    print("\n[2/3] Generating image caption on CPU...")
    start_infer = time.perf_counter()
    caption = generate_caption(processor, model, test_image)
    infer_duration = time.perf_counter() - start_infer

    print(f"-> Caption generated in {infer_duration:.2f} seconds.")
    print("-" * 65)
    print(f"Generated Caption: \"{caption}\"")
    print("-" * 65)

    assert isinstance(caption, str) and len(caption) > 0, "Generated caption must be a non-empty string!"
    print("Smoke test assertion passed: Valid non-empty caption produced.")

    # Calculate cache folder size
    print("\n[3/3] Inspecting Hugging Face cache on Drive D...")
    total_bytes = 0
    file_count = 0
    if os.path.exists(cache_dir):
        for root, _, files in os.walk(cache_dir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total_bytes += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    pass
    mb_size = total_bytes / (1024 * 1024)
    print(f"-> Total cached files: {file_count}")
    print(f"-> Total Hugging Face cache size on Drive D: {mb_size:.2f} MB")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
