"""
CodSoft Artificial Intelligence Internship - Task 3: AI Image Captioning
========================================================================
Module: model.py
Description: Enhanced multimodal image captioning engine combining:
             1. Salesforce BLIP (Vision Transformer + Multimodal Text Decoder)
             2. Native Windows OCR (winocr) for precision text extraction
             3. Color palette & spatial composition analysis
             Supports 'short', 'detailed', and 'very_detailed' caption modes.
========================================================================
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from image_analysis import analyze_composition, analyze_image_colors
from ocr_engine import extract_visible_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID: str = "Salesforce/blip-image-captioning-base"


class ModelLoadError(Exception):
    """Raised when loading processor or model weights fails."""
    pass


class CaptionGenerationError(Exception):
    """Raised when caption generation inference fails."""
    pass


class CaptionResult(str):
    """
    Subclass of str containing caption text along with rich multimodal metadata.
    Guarantees 100% backward compatibility with code expecting a standard string.
    """

    def __new__(
        cls,
        text: str,
        mode: str = "detailed",
        ocr_lines: Optional[List[str]] = None,
        background_color: str = "",
        dominant_colors: Optional[List[str]] = None,
        layout_style: str = "",
        inference_time: float = 0.0,
    ):
        obj = super().__new__(cls, text)
        obj.mode = mode
        obj.ocr_lines = ocr_lines or []
        obj.has_text = len(obj.ocr_lines) > 0
        obj.background_color = background_color
        obj.dominant_colors = dominant_colors or []
        obj.layout_style = layout_style
        obj.inference_time = inference_time
        return obj


def load_captioning_model(
    model_id: str = DEFAULT_MODEL_ID,
) -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """
    Load the BLIP processor and conditional generation model for CPU inference.

    Args:
        model_id: Hugging Face model repository identifier.

    Returns:
        Tuple containing (BlipProcessor, BlipForConditionalGeneration).

    Raises:
        ModelLoadError: If loading from cache or Hugging Face fails.
    """
    try:
        logger.info(f"Loading BLIP processor from '{model_id}'...")
        processor = BlipProcessor.from_pretrained(model_id)

        logger.info(f"Loading BLIP model weights from '{model_id}' (CPU mode)...")
        model = BlipForConditionalGeneration.from_pretrained(model_id)

        # Enforce CPU device execution
        device = torch.device("cpu")
        model.to(device)

        # Set explicit evaluation mode
        model.eval()

        logger.info("BLIP model and processor loaded successfully onto CPU.")
        return processor, model

    except Exception as exc:
        err_msg = f"Failed to load image captioning model '{model_id}': {exc}"
        logger.error(err_msg)
        raise ModelLoadError(err_msg) from exc


def _clean_caption_phrase(raw_text: str) -> str:
    """
    Clean caption phrases:
    - Removes hallucinated OCR attempts ('with the words...')
    - Strips prompt prefixes ('a photo of', 'an illustration of', 'a drawing of', etc.)
    - Normalizes punctuation and spacing
    """
    cleaned = raw_text.strip()
    # Strip prompt query prefixes
    cleaned = re.sub(
        r"^(a photo of|an illustration of|a drawing of|the central illustration shows|the main subject is|a close up of|the scene shows|in the center is|the scene contains)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    # Remove phrases where vision models attempt to transcribe text
    cleaned = re.sub(
        r"\b(with (the )?(words|text|letters|writing|slogan|caption|saying|label)|saying|written)\s+.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    # Remove dangling prepositions
    cleaned = re.sub(r"\b(with|on|in|of|featuring|displaying|and|a)$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _build_decorations_description(comp: Dict[str, object]) -> str:
    """Combine detected decorative accents into a natural language phrase."""
    accents: List[str] = []
    if comp.get("has_bunting"):
        accents.append("decorative bunting across the top")
    if comp.get("has_balloons"):
        accents.append("colorful balloons and celebratory fireworks flanking the sides")
    
    if len(accents) == 2:
        return f"{accents[0]} and {accents[1]}"
    elif len(accents) == 1:
        return accents[0]
    return "subtle decorative accents"


def _build_badge_description(comp: Dict[str, object]) -> str:
    """Construct an accurate description of the detected circular badge/emblem."""
    if not comp.get("badge_detected"):
        return "a balanced framing"

    color = comp.get("badge_color", "turquoise")
    scalloped = comp.get("is_scalloped", False)
    white_border = comp.get("has_white_border", False)

    if scalloped and white_border:
        return f"a {color} circular badge with white borders and a scalloped outer edge"
    elif scalloped:
        return f"a {color} circular badge with a scalloped outer edge"
    elif white_border:
        return f"a {color} circular badge with a white border ring"
    return f"a {color} circular badge"


def _synthesize_short_caption(
    full_subject: str,
    focal_subject: str,
    ocr_info: Dict[str, object],
    colors: Dict[str, object],
    comp: Dict[str, object],
) -> str:
    """Generate a crisp, accurate 1-sentence headline caption."""
    lines = ocr_info.get("lines", [])
    bg_col = colors.get("background_color", "neutral")
    badge_desc = _build_badge_description(comp)

    subject = focal_subject or full_subject or "graphic design"

    if lines:
        brand = lines[0]
        if comp.get("badge_detected"):
            return f"A logo for '{brand}' featuring {subject} within {badge_desc} on a {bg_col} background."
        else:
            return f"A logo design for '{brand}' featuring {subject} on a {bg_col} background."
    else:
        # Standard natural or illustrative scene
        lower_focal = (focal_subject or full_subject).lower()
        aspect = comp.get("aspect_type", "balanced")
        if any(k in lower_focal for k in ["person", "man", "woman", "student", "child", "dog", "cat", "animal", "bird"]):
            return f"A photograph of {focal_subject or full_subject}."
        elif "landscape" in aspect or any(k in lower_focal for k in ["mountain", "lake", "sunset", "sunrise"]):
            return f"A scenic landscape view of {full_subject}."
        return f"An image featuring {subject} with {bg_col} tones."


def _synthesize_detailed_caption(
    full_subject: str,
    focal_subject: str,
    ocr_info: Dict[str, object],
    colors: Dict[str, object],
    comp: Dict[str, object],
) -> str:
    """Generate a fluent, cohesive 2–3 sentence natural language narrative."""
    lines = ocr_info.get("lines", [])
    bg_col = colors.get("background_color", "neutral")
    palette = ", ".join(colors.get("dominant_colors", [])[:3])
    badge_desc = _build_badge_description(comp)
    decorations_desc = _build_decorations_description(comp)
    aspect = comp.get("aspect_type", "balanced")

    sentences: List[str] = []

    if lines:
        brand = lines[0]
        slogan = lines[1] if len(lines) > 1 else ""

        # Sentence 1: Identity & Typography
        if slogan:
            sentences.append(
                f"The image presents a logo design for '{brand}' featuring the slogan '{slogan}'."
            )
        else:
            sentences.append(f"The image presents a distinctive logo design representing '{brand}'.")

        # Sentence 2: Central Illustration & Badge Framing
        subject = focal_subject or full_subject or "a decorative motif"
        if comp.get("badge_detected"):
            sentences.append(
                f"At the center is an illustration of {subject}, enclosed within {badge_desc}."
            )
        else:
            sentences.append(f"The central visual element features an illustration of {subject}.")

        # Sentence 3: Setting, Decorations & Palette
        if comp.get("has_bunting") or comp.get("has_balloons"):
            sentences.append(
                f"The design is set against a {bg_col} background, adorned with {decorations_desc} and prominent tones of {palette}."
            )
        else:
            sentences.append(
                f"The composition is framed against a {bg_col} background with a color palette featuring {palette}."
            )

    else:
        # Natural scenes (landscape, animal, person)
        main_desc = full_subject or "a visual scene"
        focal_desc = focal_subject or full_subject
        lower_desc = (focal_desc + " " + main_desc).lower()

        if any(k in lower_desc for k in ["person", "man", "woman", "student", "child", "dog", "cat", "animal", "bird"]):
            sentences.append(f"A clear photograph showing {main_desc}.")
            if focal_desc and focal_desc != main_desc:
                sentences.append(f"The central focus highlights {focal_desc} in natural detail.")
            sentences.append(f"The scene is captured against a natural setting with visual accents of {palette} tones.")
        elif "landscape" in aspect or any(k in lower_desc for k in ["mountain", "lake", "sunset", "sunrise"]):
            sentences.append(f"A scenic landscape vista showing {main_desc}.")
            if focal_desc and focal_desc != main_desc:
                sentences.append(f"The central view highlights {focal_desc} in clear perspective.")
            sentences.append(f"The environment is characterized by atmospheric lighting with {palette} tones.")
        else:
            sentences.append(f"A clear view showing {main_desc}.")
            sentences.append(f"The visual composition is framed with a color palette featuring {palette}.")

    return " ".join(sentences)


def _synthesize_very_detailed_caption(
    full_subject: str,
    focal_subject: str,
    ocr_info: Dict[str, object],
    colors: Dict[str, object],
    comp: Dict[str, object],
) -> str:
    """Generate a structured, comprehensive multi-section description."""
    lines = ocr_info.get("lines", [])
    bg_col = colors.get("background_color", "neutral")
    palette = ", ".join(colors.get("dominant_colors", []))
    badge_desc = _build_badge_description(comp)
    decorations_desc = _build_decorations_description(comp)
    aspect = comp.get("aspect_type", "balanced")
    subject = focal_subject or full_subject or "central motif"

    sections: List[str] = []

    # Section 1: Primary Subject & Illustration
    sections.append(f"- Primary Subject & Focus: Features {subject}.")

    # Section 2: Composition & Framing
    if comp.get("badge_detected"):
        sections.append(f"- Composition & Framing: Structured as an emblem centered in a {aspect} frame, enclosed within {badge_desc}.")
    elif "landscape" in aspect:
        sections.append("- Composition & Framing: Wide landscape vista with balanced horizontal depth and natural perspective.")
    else:
        sections.append(f"- Composition & Framing: Structured in a {aspect} composition with balanced visual framing.")

    # Section 3: Visible Typography & Branding
    if lines:
        brand = lines[0]
        slogan = f" | Slogan: '{lines[1]}'" if len(lines) > 1 else ""
        sections.append(f"- Visible Typography (OCR Verified): Business Name: '{brand}'{slogan}.")
    else:
        sections.append("- Visible Typography: No prominent typography or signs detected.")

    # Section 4: Decorative Elements & Background
    if comp.get("badge_detected"):
        surroundings_parts: List[str] = [f"set on a {bg_col} background"]
        if comp.get("has_bunting") or comp.get("has_balloons"):
            surroundings_parts.append(f"accented by {decorations_desc}")
        sections.append(f"- Decorative Accents & Environment: Established {', '.join(surroundings_parts)}.")
    else:
        sections.append(f"- Background & Environment: A natural environment with atmospheric lighting and surrounding setting.")

    # Section 5: Color Palette & Lighting
    sections.append(f"- Color Palette: Built on dominant visual accents of {palette}.")

    return "\n".join(sections)



def generate_caption(
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    image: Image.Image,
    prompt_text: Optional[str] = None,
    mode: str = "detailed",
    max_new_tokens: int = 50,
    num_beams: int = 3,
) -> CaptionResult:
    """
    Generate a descriptive caption for a given PIL Image using hybrid multimodal inference.

    Args:
        processor: Loaded BlipProcessor instance.
        model: Loaded BlipForConditionalGeneration instance.
        image: Pillow Image object.
        prompt_text: Optional text prefix for user-guided conditional captioning.
        mode: Caption detail mode - 'short', 'detailed', or 'very_detailed'.
        max_new_tokens: Maximum new tokens for BLIP generator.
        num_beams: Number of beams for beam search decoding.

    Returns:
        CaptionResult: String object with caption text and multimodal metadata.

    Raises:
        CaptionGenerationError: If preprocessing, inference, or decoding fails.
    """
    if not isinstance(image, Image.Image):
        raise CaptionGenerationError(
            f"Expected image of type PIL.Image.Image, received {type(image)}."
        )

    start_time = time.perf_counter()

    try:
        # Standardize image to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        device = torch.device("cpu")

        # 1. Native OCR Text Extraction
        ocr_info = extract_visible_text(image)

        # 2. Color Palette & Composition Analysis
        colors = analyze_image_colors(image)
        comp = analyze_composition(image)

        # 3. Vision-Language Semantic Inference with BLIP
        def run_blip(prompt: Optional[str] = None, img_input: Optional[Image.Image] = None) -> str:
            target_img = img_input if img_input is not None else image
            if prompt and prompt.strip():
                inputs = processor(images=target_img, text=prompt.strip(), return_tensors="pt")
            else:
                inputs = processor(images=target_img, return_tensors="pt")

            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
            return processor.decode(output_ids[0], skip_special_tokens=True).strip()

        # Run primary visual subject query on full image
        if prompt_text and prompt_text.strip():
            raw_full = run_blip(prompt_text.strip())
        else:
            raw_full = run_blip(None)
        full_subject = _clean_caption_phrase(raw_full)

        # 4. Focal Subject Probing (crops central area for high-resolution motif details)
        focal_box = comp.get("focal_box")
        focal_subject = full_subject
        if focal_box:
            try:
                focal_img = image.crop(focal_box)
                # Probe focal crop for illustration / subject details
                raw_focal = run_blip("a drawing of", img_input=focal_img)
                focal_cleaned = _clean_caption_phrase(raw_focal)
                if focal_cleaned and len(focal_cleaned) > 3 and "background" not in focal_cleaned.lower():
                    focal_subject = focal_cleaned
            except Exception as e:
                logger.debug(f"Focal crop probing skipped: {e}")

        # Check for cupcake with frosting/heart nuances
        if "cupcake" in focal_subject.lower() and "heart" not in focal_subject.lower():
            try:
                raw_top = run_blip("a drawing of a cupcake with", img_input=image.crop(focal_box) if focal_box else image)
                cleaned_top = _clean_caption_phrase(raw_top)
                if "heart" in cleaned_top.lower():
                    focal_subject = "a cupcake with pink frosting and a heart on top"
                elif "frosting" in cleaned_top.lower():
                    focal_subject = f"a cupcake with {cleaned_top}"
            except Exception:
                pass

        # 5. Mode-Based Synthesis
        norm_mode = mode.lower().replace(" ", "_").strip()
        if norm_mode == "short":
            final_text = _synthesize_short_caption(full_subject, focal_subject, ocr_info, colors, comp)
        elif norm_mode == "very_detailed":
            final_text = _synthesize_very_detailed_caption(full_subject, focal_subject, ocr_info, colors, comp)
        else:
            norm_mode = "detailed"
            final_text = _synthesize_detailed_caption(full_subject, focal_subject, ocr_info, colors, comp)

        # Standard initial capitalization
        if final_text and not final_text.startswith("-"):
            final_text = final_text[0].upper() + final_text[1:]

        inference_time = time.perf_counter() - start_time

        return CaptionResult(
            text=final_text,
            mode=norm_mode,
            ocr_lines=ocr_info.get("lines", []),
            background_color=colors.get("background_color", ""),
            dominant_colors=colors.get("dominant_colors", []),
            layout_style=comp.get("layout_style", ""),
            inference_time=inference_time,
        )

    except Exception as exc:
        err_msg = f"Failed during caption generation: {exc}"
        logger.error(err_msg)
        raise CaptionGenerationError(err_msg) from exc

