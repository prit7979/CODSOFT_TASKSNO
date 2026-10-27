"""
CodSoft Artificial Intelligence Internship - Task 3: AI Image Captioning
========================================================================
Module: ocr_engine.py
Description: Lightweight OCR integration utilizing native Windows OCR
             (winocr / Windows.Media.Ocr) for high-accuracy, zero-overhead
             text recognition on CPU without cloud APIs.
========================================================================
"""

import logging
import re
from typing import Dict, List, Optional
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)

# Check winocr availability
try:
    import winocr
    WINOCR_AVAILABLE = True
except ImportError:
    WINOCR_AVAILABLE = False
    logger.warning("winocr is not installed. OCR will operate in fallback mode.")


def clean_extracted_text(raw_lines: List[str]) -> List[str]:
    """
    Filter and clean raw OCR lines:
    - Removes replacement characters and encoding artifacts
    - Removes isolated non-alphanumeric noise characters
    - Strips excess whitespace
    - Deduplicates identical lines
    """
    cleaned: List[str] = []
    seen = set()

    for line in raw_lines:
        text = line.replace("\ufffd", "").replace("", "").strip()
        # Remove noisy leading/trailing non-alphanumeric artifacts
        text = re.sub(r"^[^\w\s\"\'\(]+|[^\w\s\"\'\)]+$", "", text).strip()
        # Collapse multi-spaces
        text = re.sub(r"\s+", " ", text)
        # Minimum meaningful length (at least 2 letters/numbers)
        if len(re.findall(r"[A-Za-z0-9]", text)) >= 2:
            norm = text.lower()
            if norm not in seen:
                seen.add(norm)
                cleaned.append(text)

    return cleaned


def extract_visible_text(image: Image.Image) -> Dict[str, object]:
    """
    Extract visible text from a PIL Image using native Windows OCR.

    Args:
        image: Pillow Image object.

    Returns:
        Dictionary with:
          - 'has_text': bool indicating if legible text was found
          - 'lines': list of clean detected text strings
          - 'primary_text': main heading or brand name (first/largest line)
          - 'secondary_text': subtitles, slogans, or auxiliary labels
          - 'full_text': formatted string combining lines
    """
    result = {
        "has_text": False,
        "lines": [],
        "primary_text": "",
        "secondary_text": "",
        "full_text": "",
    }

    if not WINOCR_AVAILABLE or not isinstance(image, Image.Image):
        return result

    try:
        # Standardize to RGB
        img_rgb = image.convert("RGB")
        w, h = img_rgb.size

        # Preprocessing: If image is small (<600px), upscale slightly for better OCR accuracy
        if max(w, h) < 600:
            scale = 600 / max(w, h)
            img_to_ocr = img_rgb.resize(
                (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
            )
        else:
            img_to_ocr = img_rgb

        # Execute native Windows OCR
        ocr_res = winocr.recognize_pil_sync(img_to_ocr)
        raw_lines = [
            line.get("text", "").strip()
            for line in ocr_res.get("lines", [])
            if line.get("text", "").strip()
        ]

        cleaned_lines = clean_extracted_text(raw_lines)

        if cleaned_lines:
            result["has_text"] = True
            result["lines"] = cleaned_lines
            result["primary_text"] = cleaned_lines[0]
            result["secondary_text"] = " ".join(cleaned_lines[1:]) if len(cleaned_lines) > 1 else ""
            result["full_text"] = " | ".join(cleaned_lines)

    except Exception as exc:
        logger.warning(f"Native OCR extraction encountered an error: {exc}")

    return result
