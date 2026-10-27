"""
CodSoft Artificial Intelligence Internship - Task 3: AI Image Captioning
========================================================================
Module: image_analysis.py
Description: Fast, deterministic visual analysis for color palettes,
             background tones, badge geometry, scalloped edges,
             decorative festive accents, and spatial layout composition.
========================================================================
"""

import colorsys
import math
from typing import Dict, List, Optional, Tuple
from PIL import Image

# Enhanced reference color centroids with calibrated perceptual anchors
COLOR_PALETTE_REFERENCE: List[Tuple[str, Tuple[int, int, int]]] = [
    ("light pink", (255, 228, 235)),
    ("soft pink", (255, 182, 193)),
    ("pink", (255, 105, 180)),
    ("deep red", (180, 20, 30)),
    ("crimson", (220, 20, 60)),
    ("turquoise / teal", (64, 224, 208)),
    ("teal", (0, 128, 128)),
    ("cyan / light blue", (135, 206, 235)),
    ("royal blue", (65, 105, 225)),
    ("steel blue", (70, 130, 180)),
    ("dark navy", (20, 30, 70)),
    ("midnight blue", (25, 25, 112)),
    ("vibrant green", (34, 139, 34)),
    ("emerald green", (46, 139, 87)),
    ("lime green", (50, 205, 50)),
    ("olive green", (107, 142, 35)),
    ("warm gold / yellow", (255, 215, 0)),
    ("soft yellow", (255, 245, 160)),
    ("orange", (255, 140, 0)),
    ("purple / violet", (147, 112, 219)),
    ("wooden brown", (139, 90, 43)),
    ("tan / beige", (210, 180, 140)),
    ("pure white", (255, 255, 255)),
    ("neutral gray", (128, 128, 128)),
    ("charcoal / dark gray", (50, 50, 50)),
    ("black", (15, 15, 15)),
]


def rgb_to_color_name(r: int, g: int, b: int) -> str:
    """Find the closest descriptive color name using weighted Euclidean distance in RGB space."""
    best_name = "neutral"
    min_dist = float("inf")

    for name, ref_rgb in COLOR_PALETTE_REFERENCE:
        # Perceptual weighting: red=0.3, green=0.59, blue=0.11
        dr = r - ref_rgb[0]
        dg = g - ref_rgb[1]
        db = b - ref_rgb[2]
        dist = 0.3 * (dr ** 2) + 0.59 * (dg ** 2) + 0.11 * (db ** 2)
        if dist < min_dist:
            min_dist = dist
            best_name = name

    return best_name


def analyze_image_colors(image: Image.Image) -> Dict[str, object]:
    """
    Extract the background color and top dominant colors from a PIL Image.

    Returns:
        Dictionary with:
          - 'background_color': string name of background tone
          - 'background_rgb': (r, g, b) tuple
          - 'dominant_colors': list of top prominent named colors
    """
    rgb_img = image.convert("RGB")
    w, h = rgb_img.size

    # Sample outer perimeter pixels to establish true background tone
    sample_points = (
        [(x, 0) for x in range(0, w, max(1, w // 20))]
        + [(x, h - 1) for x in range(0, w, max(1, w // 20))]
        + [(0, y) for y in range(0, h, max(1, h // 20))]
        + [(w - 1, y) for y in range(0, h, max(1, h // 20))]
    )
    bg_rgbs = [rgb_img.getpixel(pt) for pt in sample_points]
    avg_bg_r = sum(c[0] for c in bg_rgbs) // len(bg_rgbs)
    avg_bg_g = sum(c[1] for c in bg_rgbs) // len(bg_rgbs)
    avg_bg_b = sum(c[2] for c in bg_rgbs) // len(bg_rgbs)
    background_name = rgb_to_color_name(avg_bg_r, avg_bg_g, avg_bg_b)

    # Extract dominant palette by resizing and quantizing
    thumb = rgb_img.resize((120, 120), Image.Resampling.BOX)
    quantized = thumb.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()

    dominant_names: List[str] = []
    seen = {background_name}  # Avoid repeating background as first dominant

    if palette:
        for i in range(min(7, len(palette) // 3)):
            r, g, b = palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]
            c_name = rgb_to_color_name(r, g, b)
            if c_name not in seen:
                seen.add(c_name)
                dominant_names.append(c_name)

    return {
        "background_color": background_name,
        "background_rgb": (avg_bg_r, avg_bg_g, avg_bg_b),
        "dominant_colors": dominant_names[:4],
    }


def analyze_composition(image: Image.Image) -> Dict[str, object]:
    """
    Comprehensive visual analysis detecting:
    - Aspect ratio classification
    - Central circular badge or emblem
    - Scalloped outer borders and concentric border rings
    - Decorative party bunting / garland across the top
    - Flanking celebratory balloons or sparkles on the margins
    - Recommended focal crop coordinates for detailed semantic probing

    Returns:
        Dictionary with structural layout, features, and crop bounds.
    """
    rgb_img = image.convert("RGB")
    w, h = rgb_img.size
    cx, cy = w // 2, h // 2
    aspect_ratio = w / max(h, 1)

    if aspect_ratio > 1.25:
        aspect_type = "landscape"
    elif aspect_ratio < 0.8:
        aspect_type = "portrait"
    else:
        aspect_type = "square / balanced"

    # 1. Background sampling and corner uniformity
    corners = [
        rgb_img.getpixel((5, 5)),
        rgb_img.getpixel((w - 6, 5)),
        rgb_img.getpixel((5, h - 6)),
        rgb_img.getpixel((w - 6, h - 6)),
    ]
    corner_lums = [sum(c) for c in corners]
    corner_variance = max(corner_lums) - min(corner_lums)
    is_solid_backdrop = corner_variance < 50

    perim = [rgb_img.getpixel((x, 0)) for x in range(0, w, max(1, w // 25))] + [
        rgb_img.getpixel((0, y)) for y in range(0, h, max(1, h // 25))
    ]
    avg_bg = (
        sum(c[0] for c in perim) // len(perim),
        sum(c[1] for c in perim) // len(perim),
        sum(c[2] for c in perim) // len(perim),
    )

    # 2. Radial badge detection (only valid on graphic designs with uniform backdrop)
    max_radius = min(cx, cy)
    badge_detected = False
    badge_radius = 0
    badge_color_name = ""
    has_white_border = False
    is_scalloped = False
    has_bunting = False
    has_balloons = False

    if is_solid_backdrop:
        step_r = max(4, max_radius // 60)
        radial_profile: List[Tuple[int, Tuple[int, int, int]]] = []

        for r in range(step_r * 2, max_radius - step_r, step_r):
            ring_samples = []
            for angle_deg in range(0, 360, 20):
                rad = math.radians(angle_deg)
                x = int(cx + r * math.cos(rad))
                y = int(cy + r * math.sin(rad))
                if 0 <= x < w and 0 <= y < h:
                    ring_samples.append(rgb_img.getpixel((x, y)))
            if ring_samples:
                avg_ring = (
                    sum(s[0] for s in ring_samples) // len(ring_samples),
                    sum(s[1] for s in ring_samples) // len(ring_samples),
                    sum(s[2] for s in ring_samples) // len(ring_samples),
                )
                radial_profile.append((r, avg_ring))

        # Detect boundary where contrast against background drops sharply
        for i in range(len(radial_profile) - 1, 2, -1):
            r, col = radial_profile[i]
            diff_bg = math.sqrt(sum((c - b) ** 2 for c, b in zip(col, avg_bg)))
            if diff_bg > 55:
                badge_detected = True
                badge_radius = r
                # Sample interior core for true badge color
                core_r = max(10, int(r * 0.75))
                core_samples = [
                    rgb_img.getpixel((int(cx + core_r * math.cos(math.radians(a))), int(cy + core_r * math.sin(math.radians(a)))))
                    for a in range(0, 360, 20)
                ]
                if core_samples:
                    avg_c = (
                        sum(s[0] for s in core_samples) // len(core_samples),
                        sum(s[1] for s in core_samples) // len(core_samples),
                        sum(s[2] for s in core_samples) // len(core_samples),
                    )
                    badge_color_name = rgb_to_color_name(avg_c[0], avg_c[1], avg_c[2])
                else:
                    badge_color_name = rgb_to_color_name(col[0], col[1], col[2])

                # Check for white border ring
                for j in range(max(0, i - 8), i):
                    r_check, _ = radial_profile[j]
                    samples_ring = [
                        rgb_img.getpixel((int(cx + r_check * math.cos(math.radians(a))), int(cy + r_check * math.sin(math.radians(a)))))
                        for a in range(0, 360, 10)
                    ]
                    white_pct = sum(1 for s in samples_ring if s[0] > 235 and s[1] > 235 and s[2] > 235) / max(1, len(samples_ring))
                    if white_pct > 0.35:
                        has_white_border = True
                        break
                break

        # If badge detected, test for scalloped / undulating outer edge
        if badge_detected and badge_radius > 40:
            circ_samples = []
            for angle_deg in range(0, 360, 3):
                rad = math.radians(angle_deg)
                x = int(cx + (badge_radius + 4) * math.cos(rad))
                y = int(cy + (badge_radius + 4) * math.sin(rad))
                if 0 <= x < w and 0 <= y < h:
                    circ_samples.append(rgb_img.getpixel((x, y)))
            if circ_samples:
                lums = [0.299 * s[0] + 0.587 * s[1] + 0.114 * s[2] for s in circ_samples]
                diffs = sum(abs(lums[k] - lums[k - 1]) for k in range(1, len(lums))) / len(lums)
                if diffs > 8.0:
                    is_scalloped = True

        # Decorative top bunting / banner detection
        top_y = int(h * 0.05)
        top_pixels = [rgb_img.getpixel((x, top_y)) for x in range(int(w * 0.1), int(w * 0.9), max(1, w // 40))]
        top_var = max(sum(p) for p in top_pixels) - min(sum(p) for p in top_pixels) if top_pixels else 0
        has_bunting = top_var > 140

        # Flanking celebration balloons
        left_x = int(w * 0.12)
        right_x = int(w * 0.88)
        left_pixels = [rgb_img.getpixel((left_x, y)) for y in range(int(h * 0.2), int(h * 0.6), max(1, h // 25))]
        right_pixels = [rgb_img.getpixel((right_x, y)) for y in range(int(h * 0.2), int(h * 0.6), max(1, h // 25))]
        left_diffs = [math.sqrt(sum((c - b) ** 2 for c, b in zip(p, avg_bg))) for p in left_pixels]
        right_diffs = [math.sqrt(sum((c - b) ** 2 for c, b in zip(p, avg_bg))) for p in right_pixels]
        has_balloons = (sum(1 for d in left_diffs if d > 50) >= 2) or (sum(1 for d in right_diffs if d > 50) >= 2)

    # 5. Determine recommended focal crop
    if badge_detected and badge_radius > 50:
        focal_box = (
            max(0, cx - int(badge_radius * 0.65)),
            max(0, cy - int(badge_radius * 0.75)),
            min(w, cx + int(badge_radius * 0.65)),
            min(h, cy + int(badge_radius * 0.25)),
        )
    else:
        focal_box = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))

    # Assemble descriptive visual features
    features: List[str] = []
    if badge_detected:
        badge_desc_parts = [badge_color_name]
        if is_scalloped:
            badge_desc_parts.append("scalloped outer edge")
        if has_white_border:
            badge_desc_parts.append("white circular border")
        badge_desc_str = f"a {badge_color_name} circular badge"
        if is_scalloped and has_white_border:
            badge_desc_str += " with white borders and a scalloped outer edge"
        elif is_scalloped:
            badge_desc_str += " with a scalloped outer edge"
        elif has_white_border:
            badge_desc_str += " with white borders"
        features.append(badge_desc_str)
    elif aspect_type == "landscape":
        features.append("a wide landscape panorama")
    else:
        features.append("a balanced central composition")

    if has_bunting:
        features.append("decorative bunting across the top")
    if has_balloons:
        features.append("colorful balloons and celebratory accents flanking the sides")

    return {
        "aspect_type": aspect_type,
        "badge_detected": badge_detected,
        "badge_color": badge_color_name,
        "has_white_border": has_white_border,
        "is_scalloped": is_scalloped,
        "has_bunting": has_bunting,
        "has_balloons": has_balloons,
        "focal_box": focal_box,
        "features": features,
        "layout_style": "circular emblem / badge design" if badge_detected else f"{aspect_type} visual scene",
    }

