"""
Streamlit Web Application for Face Detection using OpenCV Haar Cascades.

This interactive app supports both file uploads (JPG, JPEG, PNG) and real-time
webcam snapshots. It allows users to adjust detection sensitivity (scale factor
and minimum neighbors), visualizes detected bounding boxes, reports face counts,
and provides a one-click PNG download of processed results.
"""

import io
from typing import Optional
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

from detector import FaceDetector


@st.cache_resource
def get_detector() -> FaceDetector:
    """
    Initializes and caches the FaceDetector instance across user interactions
    to prevent redundant re-parsing and re-loading of the Haar Cascade XML model.
    """
    return FaceDetector()


def load_image_as_rgb(file_source: io.BytesIO) -> np.ndarray:
    """
    Loads an uploaded or captured image file into a standardized 3-channel RGB
    NumPy array, handling color-space conversions and EXIF orientations.

    Args:
        file_source: In-memory byte buffer containing image data.

    Returns:
        np.ndarray: uint8 image array with shape (height, width, 3) in RGB order.

    Raises:
        ValueError: If the file cannot be decoded as a valid image.
    """
    try:
        pil_img = Image.open(file_source)
        # Transpose image according to EXIF orientation tag if present
        pil_img = ImageOps.exif_transpose(pil_img)
        # Convert any mode (RGBA, grayscale, palette) to standard 3-channel RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img)
    except Exception as exc:
        raise ValueError(f"Unable to decode image file: {exc}") from exc


def main() -> None:
    st.set_page_config(
        page_title="Face Detection App | OpenCV & Streamlit",
        page_icon="👤",
        layout="wide",
    )

    st.title("👤 Real-Time Face Detection")
    st.markdown(
        "Detect and count human faces in photos or webcam snapshots using "
        "OpenCV's pre-trained **YuNet deep learning neural network**."
    )
    st.divider()

    # Sidebar: Detection Hyperparameters
    st.sidebar.header("⚙️ Detection Settings")
    st.sidebar.markdown(
        "Tune these parameters to balance detection recall and precision."
    )

    score_threshold = st.sidebar.slider(
        "Confidence Threshold (`scoreThreshold`)",
        min_value=0.30,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help=(
            "Minimum detection confidence score. Higher values eliminate false positives "
            "and background noise; lower values increase sensitivity for subtle or distant faces."
        ),
    )

    nms_threshold = st.sidebar.slider(
        "NMS IoU Threshold (`nmsThreshold`)",
        min_value=0.10,
        max_value=0.70,
        value=0.30,
        step=0.05,
        help=(
            "Non-Maximum Suppression Intersection-over-Union threshold. "
            "Lower values aggressively merge and suppress overlapping bounding boxes for the same face."
        ),
    )

    st.sidebar.subheader("🎨 Annotation Style")
    box_color_choice = st.sidebar.selectbox(
        "Bounding Box Color",
        options=["Green", "Cyan", "Red", "Yellow", "Magenta"],
        index=0,
    )
    color_map = {
        "Green": (0, 255, 0),
        "Cyan": (0, 255, 255),
        "Red": (255, 0, 0),
        "Yellow": (255, 255, 0),
        "Magenta": (255, 0, 255),
    }
    box_color = color_map.get(box_color_choice, (0, 255, 0))

    box_thickness = st.sidebar.slider(
        "Box Thickness",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
    )

    draw_labels = st.sidebar.checkbox(
        "Display Face Index Labels (e.g., 'Face #1')",
        value=True,
    )

    # Initialize FaceDetector
    try:
        detector = get_detector()
    except Exception as exc:
        st.error(f"❌ Failed to initialize FaceDetector: {exc}")
        st.stop()

    # Image Input Selection
    st.subheader("1. Select Image Source")
    input_mode = st.radio(
        "Choose how you want to provide an image:",
        options=["📁 Upload an Image", "📸 Use Webcam Capture"],
        horizontal=True,
    )

    image_source: Optional[io.BytesIO] = None

    if input_mode == "📁 Upload an Image":
        uploaded_file = st.file_uploader(
            "Upload an image file (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: .jpg, .jpeg, .png",
        )
        if uploaded_file is not None:
            image_source = uploaded_file
    else:
        camera_snapshot = st.camera_input("Take a snapshot with your webcam")
        if camera_snapshot is not None:
            image_source = camera_snapshot

    if image_source is None:
        st.info("👆 Please upload an image or capture a webcam photo to begin face detection.")
        return

    # Image Processing & Face Detection
    try:
        rgb_image = load_image_as_rgb(image_source)
    except Exception as err:
        st.error(f"❌ Error loading image: {err}")
        return

    try:
        faces = detector.detect_faces(
            image=rgb_image,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            is_rgb=True,
        )
        face_count = len(faces)
        annotated_image = detector.draw_faces(
            image=rgb_image,
            faces=faces,
            color=box_color,
            thickness=box_thickness,
            draw_labels=draw_labels,
        )
    except Exception as err:
        st.error(f"❌ Face detection failed: {err}")
        return

    # Results Section
    st.divider()
    st.subheader("2. Detection Results")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Total Faces Detected", face_count)
    with metric_col2:
        st.metric("Image Dimensions", f"{rgb_image.shape[1]} × {rgb_image.shape[0]} px")
    with metric_col3:
        detection_status = "✅ Face(s) Found" if face_count > 0 else "⚠️ No Faces Detected"
        st.metric("Detection Status", detection_status)

    if face_count == 0:
        st.warning(
            "No faces detected with current settings. Try lowering the 'Confidence Threshold' "
            "in the sidebar if faces are visible."
        )

    # Side-by-side Image Comparison
    view_col1, view_col2 = st.columns(2)
    with view_col1:
        st.markdown("**Original Image**")
        st.image(rgb_image, caption="Original Input", use_container_width=True)

    with view_col2:
        st.markdown("**Processed Image**")
        st.image(
            annotated_image,
            caption=f"Annotated Image ({face_count} face{'s' if face_count != 1 else ''} detected)",
            use_container_width=True,
        )

    # Download Button
    st.divider()
    st.subheader("3. Export Processed Image")
    try:
        buf = io.BytesIO()
        annotated_pil = Image.fromarray(annotated_image)
        annotated_pil.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        st.download_button(
            label="⬇️ Download Processed Image (PNG)",
            data=png_bytes,
            file_name="detected_faces.png",
            mime="image/png",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Failed to generate downloadable image: {exc}")


if __name__ == "__main__":
    main()
