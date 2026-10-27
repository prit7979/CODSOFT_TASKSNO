"""
CodSoft Artificial Intelligence Internship - Task 3: AI Image Captioning
========================================================================
Module: app.py
Description: Interactive Streamlit web application providing intelligent,
             detailed image captioning via Salesforce BLIP, Native Windows OCR,
             and computer vision color/composition analysis.
========================================================================
"""

import io
import os
import sys
import time
from typing import Optional, Tuple

import streamlit as st
from PIL import Image

# Ensure module path resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    CaptionGenerationError,
    CaptionResult,
    ModelLoadError,
    generate_caption,
    load_captioning_model,
)

# Page Configuration
st.set_page_config(
    page_title="Intelligent AI Image Captioning | CodSoft AI",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_model() -> Tuple[object, object]:
    """
    Load and cache the BLIP processor and model in Streamlit application state.
    Cached so weights remain in memory across browser interactions.
    """
    return load_captioning_model()


def validate_and_open_image(uploaded_file) -> Optional[Image.Image]:
    """
    Safely validate and open an uploaded image stream with Pillow.

    Returns:
        PIL.Image.Image if valid, None if invalid or corrupt.
    """
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Validate image integrity without full decoding
        # Reopen for actual rendering/inference since verify consumes stream
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as exc:
        st.error(f"⚠️ Invalid image file: The uploaded file could not be decoded. ({exc})")
        return None


def main() -> None:
    # Header Section
    st.title("🖼️ Intelligent AI Image Captioning")
    st.caption("CodSoft Artificial Intelligence Internship — Task 3")

    st.markdown(
        """
        Generate **specific, accurate, and detailed natural language descriptions**
        of any image. Powered by a hybrid multimodal engine combining **Salesforce BLIP**,
        **Native Windows OCR** for visible text, and **color/composition analysis**.
        
        *Runs 100% locally on CPU — completely private with zero cloud uploads.*
        """
    )

    # Sidebar: Controls & Options
    with st.sidebar:
        st.header("⚙️ Caption Settings")

        # Caption Detail Mode Selector
        detail_mode = st.radio(
            "Select Caption Detail Level:",
            options=["Short", "Detailed", "Very Detailed"],
            index=1,
            help=(
                "• Short: Crisp 1-sentence headline.\n"
                "• Detailed: Rich 2-3 sentence narrative describing subject, layout, and colors.\n"
                "• Very Detailed: Comprehensive structured breakdown of all visual attributes."
            ),
        )

        st.divider()

        # Optional Prompt Prefix for Guided Captioning
        st.subheader("💡 Optional Prompt Prefix")
        prompt_prefix = st.text_input(
            "Guide caption with a prefix (optional):",
            placeholder="e.g. 'A professional photo of' or leave blank",
            help="Leave blank for unconditional captioning, or provide a prompt to guide generation.",
        )

        st.divider()
        st.header("🔬 Engine Specifications")
        st.markdown(
            """
            - **Vision Model:** `Salesforce/blip-image-captioning-base` (~224M params)
            - **OCR Engine:** Native Windows Media OCR (Zero cloud dependencies)
            - **Inference Mode:** CPU (`torch.inference_mode`)
            - **Hardware Compatibility:** Tested on 8 GB RAM Windows systems
            """
        )
        st.caption("Developed by Prit Panvalkar / CodSoft AI Intern")

    # Image Input Tabs: File Upload or Sample Benchmark Images
    input_tab1, input_tab2 = st.tabs(["📤 Upload Image", "🎨 Sample Benchmark Images"])

    selected_image: Optional[Image.Image] = None
    image_filename: str = "captioned_image.jpg"

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image:",
            type=["jpg", "jpeg", "png"],
            help="Maximum recommended resolution: 1080p for fast CPU inference.",
        )
        if uploaded_file is not None:
            selected_image = validate_and_open_image(uploaded_file)
            image_filename = uploaded_file.name

    with input_tab2:
        st.markdown("**Select a built-in benchmark image for instant testing:**")
        benchmarks_dir = os.path.join(os.path.dirname(__file__), "benchmark_samples")

        sample_options = {
            "🧁 Bakery Logo (SupriLicious Cakes)": os.path.join(benchmarks_dir, "bakery_logo.png"),
            "🏔️ Scenic Landscape Vista": os.path.join(benchmarks_dir, "landscape_sample.jpg"),
            "🐕 Golden Dog in Grassland": os.path.join(benchmarks_dir, "animal_sample.jpg"),
            "💻 Person Working at Desk": os.path.join(benchmarks_dir, "person_sample.jpg"),
            "🏠 Basic Geometric House": os.path.join(os.path.dirname(__file__), "sample_test_image.jpg"),
        }

        chosen_sample = st.selectbox("Choose a sample scene:", list(sample_options.keys()))
        sample_path = sample_options[chosen_sample]

        if os.path.exists(sample_path):
            if st.button("Load This Sample", use_container_width=True):
                try:
                    selected_image = Image.open(sample_path)
                    image_filename = os.path.basename(sample_path)
                except Exception as e:
                    st.error(f"Could not load sample image: {e}")
        else:
            st.warning("Benchmark sample not found on disk.")

    # Image Preview & Caption Generation Section
    if selected_image is not None:
        st.divider()
        st.subheader("📸 Image Preview")
        st.image(
            selected_image,
            caption=f"Selected: {image_filename} ({selected_image.size[0]}×{selected_image.size[1]} px)",
            use_container_width=True,
        )

        # Action Button
        generate_clicked = st.button(
            f"✨ Generate {detail_mode} Caption",
            type="primary",
            use_container_width=True,
        )

        if generate_clicked:
            # 1. Ensure Model is Loaded
            try:
                with st.spinner("⏳ Loading BLIP vision model into CPU memory..."):
                    processor, model = get_model()
            except ModelLoadError as err:
                st.error(f"❌ Failed to load model weights: {err}")
                return
            except Exception as err:
                st.error(f"❌ Unexpected error while loading model: {err}")
                return

            # 2. Perform Inference
            start_time = time.perf_counter()
            try:
                with st.spinner(f"🧠 Analyzing visual features, text, and colors ({detail_mode} Mode)..."):
                    result: CaptionResult = generate_caption(
                        processor=processor,
                        model=model,
                        image=selected_image,
                        prompt_text=prompt_prefix if prompt_prefix else None,
                        mode=detail_mode.lower(),
                    )
                inference_time = time.perf_counter() - start_time

                # 3. Display Results
                st.success(f"🎉 {detail_mode} Caption Generated Successfully!")
                st.markdown(f"### 📝 {detail_mode} Description:")

                if detail_mode == "Very Detailed":
                    st.markdown(result)
                else:
                    st.info(f"**\"{result}\"**")

                # 4. Display OCR Text Card if Text was Detected
                if hasattr(result, "has_text") and result.has_text:
                    st.markdown("#### 🔤 Detected Visible Text (OCR Verified):")
                    for i, line in enumerate(result.ocr_lines, 1):
                        st.markdown(f"- **Line {i}:** `{line}`")

                # 5. Visual Attributes Metadata
                col1, col2 = st.columns(2)
                with col1:
                    bg = getattr(result, "background_color", "N/A")
                    st.caption(f"🎨 **Background Tone:** {bg}")
                with col2:
                    pal = ", ".join(getattr(result, "dominant_colors", [])) or "N/A"
                    st.caption(f"🌈 **Key Palette:** {pal}")

                st.caption(f"⏱️ CPU Processing Latency: **{inference_time:.2f} seconds**")

                # 6. Download Caption Button
                txt_filename = f"{os.path.splitext(image_filename)[0]}_{detail_mode.lower()}_caption.txt"
                download_content = (
                    f"Image: {image_filename}\n"
                    f"Detail Mode: {detail_mode}\n"
                    f"Inference Latency: {inference_time:.2f}s (CPU)\n"
                    f"Model: Salesforce/blip-image-captioning-base + Native Windows OCR\n"
                    f"{'=' * 50}\n"
                    f"Generated Caption:\n{result}\n"
                )
                if hasattr(result, "has_text") and result.has_text:
                    download_content += f"\nVisible Text:\n" + "\n".join(f"- {l}" for l in result.ocr_lines) + "\n"

                st.download_button(
                    label=f"💾 Download {detail_mode} Caption as .txt",
                    data=download_content,
                    file_name=txt_filename,
                    mime="text/plain",
                    use_container_width=True,
                )

            except CaptionGenerationError as err:
                st.error(f"❌ Caption generation error: {err}")
            except Exception as err:
                st.error(f"❌ Unexpected error during inference: {err}")

    else:
        st.info("👆 Upload an image or select a benchmark sample above to generate an intelligent caption.")


if __name__ == "__main__":
    main()
