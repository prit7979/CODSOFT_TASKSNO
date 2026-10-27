"""
Comprehensive Benchmark Runner for Task 3 AI Image Captioning
==============================================================
Evaluates:
1. Bakery Logo (SupriLicious Cakes) - Logo with text, slogan, badge, cupcake, balloons
2. Landscape - Mountain range, lake, sunset
3. Animal - Dog standing in grassland
4. Person - Person working at desk with laptop

Measures:
- Model load time
- RAM working set (MB)
- Inference latency per image
- Generated captions across Short, Detailed, and Very Detailed modes
"""

import os
import subprocess
import sys
import time
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import load_captioning_model, generate_caption


def get_ram_mb() -> float:
    try:
        cmd = ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {os.getpid()}).WorkingSet64 / 1MB"]
        res = subprocess.check_output(cmd).decode().strip()
        return float(res)
    except Exception:
        return 0.0


def main():
    print("=" * 70)
    print("  CodSoft AI Internship - Task 3: Comprehensive Benchmarking Suite")
    print("=" * 70)

    ram_start = get_ram_mb()
    print(f"Initial Process RAM: {ram_start:.2f} MB")

    print("\n[Step 1] Loading BLIP vision model into memory (CPU)...")
    start_load = time.perf_counter()
    processor, model = load_captioning_model()
    load_time = time.perf_counter() - start_load
    ram_after_load = get_ram_mb()

    print(f"-> Model Loaded in: {load_time:.2f} seconds")
    print(f"-> RAM after model load: {ram_after_load:.2f} MB (Delta: +{ram_after_load - ram_start:.2f} MB)")

    benchmarks_dir = os.path.join(os.path.dirname(__file__), "benchmark_samples")
    test_cases = [
        ("1. Bakery Logo (SupriLicious Cakes)", os.path.join(benchmarks_dir, "bakery_logo.png")),
        ("2. Landscape Vista", os.path.join(benchmarks_dir, "landscape_sample.jpg")),
        ("3. Animal in Nature (Dog)", os.path.join(benchmarks_dir, "animal_sample.jpg")),
        ("4. Person at Workplace", os.path.join(benchmarks_dir, "person_sample.jpg")),
    ]

    print("\n[Step 2] Executing Multi-Mode Captioning across Benchmark Images...")

    total_infer_time = 0.0

    for title, img_path in test_cases:
        print("\n" + "#" * 70)
        print(f"TEST CASE: {title}")
        print("#" * 70)

        if not os.path.exists(img_path):
            print(f"ERROR: Image path {img_path} not found!")
            continue

        img = Image.open(img_path)

        # 1. Short Mode
        t0 = time.perf_counter()
        short_cap = generate_caption(processor, model, img, mode="short")
        t_short = time.perf_counter() - t0

        # 2. Detailed Mode
        t1 = time.perf_counter()
        det_cap = generate_caption(processor, model, img, mode="detailed")
        t_det = time.perf_counter() - t1

        # 3. Very Detailed Mode
        t2 = time.perf_counter()
        vdet_cap = generate_caption(processor, model, img, mode="very_detailed")
        t_vdet = time.perf_counter() - t2

        total_infer_time += (t_short + t_det + t_vdet)

        print(f"\n[SHORT MODE] (Latency: {t_short:.2f}s):")
        print(f"\"{short_cap}\"")

        print(f"\n[DETAILED MODE] (Latency: {t_det:.2f}s):")
        print(f"\"{det_cap}\"")

        print(f"\n[VERY DETAILED MODE] (Latency: {t_vdet:.2f}s):")
        print(f"{vdet_cap}")

        if hasattr(det_cap, "has_text") and det_cap.has_text:
            print(f"\n[OCR EXTRACTED TEXT]: {det_cap.ocr_lines}")
        else:
            print(f"\n[OCR EXTRACTED TEXT]: None (Visual-only scene)")

        print(f"[ATTRIBUTES]: Background: {det_cap.background_color} | Palette: {', '.join(det_cap.dominant_colors)}")

    ram_final = get_ram_mb()
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Model Load Time (Cached on Drive D): {load_time:.2f} s")
    print(f"Peak Working Set Memory: {ram_final:.2f} MB (~{ram_final / 1024:.2f} GB)")
    print(f"Total Inference Time across 12 runs (4 images x 3 modes): {total_infer_time:.2f} s")
    print(f"Average Inference Time per Request: {total_infer_time / 12:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
