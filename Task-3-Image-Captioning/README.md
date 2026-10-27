# CodSoft Artificial Intelligence Internship - Task 3: Intelligent AI Image Captioning

An upgraded, privacy-preserving AI image captioning system that produces **specific, detailed, and accurate descriptions** of images. Combines **Salesforce BLIP** (*Vision Transformer + Multimodal Text Decoder*), **Native Windows OCR** (`winocr` / `Windows.Media.Ocr`), and **computer vision color & composition analysis** to deliver multi-mode captions running **100% locally on CPU**.

---

## 📌 Multimodal Architecture & Design Rationale

Standard vision-language models often produce generic, 5-to-7-word captions (e.g., classifying a detailed bakery logo simply as *"Cupcakes love and passion"*). To solve this without exhausting the memory limits of an 8 GB RAM laptop, this project employs a **hybrid multimodal pipeline**:

```text
┌─────────────────┐     ┌─────────────────────────────────────────────────────────┐
│                 │ ──> │ Native Windows OCR (winocr / Windows.Media.Ocr)         │ ──> Exact Brand & Slogans
│                 │     └─────────────────────────────────────────────────────────┘
│  Uploaded Image │ ──> │ Color & Composition Engine (image_analysis.py)          │ ──> Palette, Background, Layout
│   (JPG/PNG)     │     └─────────────────────────────────────────────────────────┘
│                 │ ──> │ Salesforce BLIP ViT-B/16 (model.py)                     │ ──> Semantic Subjects & Scene
└─────────────────┘     └─────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │ Context-Aware Caption Synthesizer (model.py)            │
                        │ Modes: Short | Detailed | Very Detailed                 │
                        └─────────────────────────────────────────────────────────┘
```

### 1. Vision Model: Salesforce BLIP (`Salesforce/blip-image-captioning-base`)
- **Parameters:** ~224M parameters.
- **Why Selected:** Proven compatibility with Python 3.13 on Windows CPU; peak working set is only **~1.40 GB RAM**, safely fitting within an 8 GB RAM budget without causing memory thrashing. Heavier models (e.g., Moondream 1.8B or Florence-2) either exhibit severe remote-code incompatibilities with `transformers 5.x` on Python 3.13 or exceed CPU interactive latency limits.
- **Execution:** Runs in `torch.inference_mode()` with `num_beams=3`.

### 2. OCR Engine: Native Windows Media OCR (`winocr`)
- **Why Selected:** Built directly into Windows 10/11 (`Windows.Media.Ocr`).
- **Zero Cloud / Zero Overhead:** Requires **0 MB** of model downloads (less than 350 KB wrapper), processes text in milliseconds on CPU, and extracts business names, slogans, and labels with near-perfect precision without guessing or hallucinating words.

### 3. Color & Layout Engine (`image_analysis.py`)
- Analyzes perimeter pixels to accurately identify the background tone (e.g. *light pink*, *turquoise*, *neutral gray*).
- Extracts dominant RGB palettes and maps them to descriptive human color names.
- Determines geometric composition (e.g. *central badge/emblem*, *scenic landscape vista*, *focused portrait*).

---

## ✨ Caption Modes Supported

Users can toggle between three caption detail levels:

1. **Short Mode:**
   A crisp, single-sentence headline identifying the entity/scene, main visual subject, and key brand name.
   - *Example:* `"A logo design for 'SupriLicious Cakes' featuring a cupcake on a light pink background."*

2. **Detailed Mode:**
   A cohesive 2–3 sentence natural language narrative detailing the central subject, visual layout, background, colors, and embedded text.
   - *Example:* `"A decorative design for 'SupriLicious Cakes' featuring the slogan 'Custom Cakes made with Love and Passion'. The central visual element features a cupcake, structured in a focal subject composition. The setting features a cupcake and balloons, set against a light pink background with tones of light pink, cyan / light blue, turquoise / teal."*

3. **Very Detailed Mode:**
   A comprehensive, structured multi-section breakdown:
   - `- Primary Subject & Focus:` Central objects and characters.
   - `- Composition & Framing:` Geometric structure (badge, vista, portrait).
   - `- Color Palette:` Background foundation and dominant accents.
   - `- Background & Environment:` Surrounding objects and decorative accents.
   - `- Visible Text (OCR Verified):` Verified text, slogans, and labels.

---

## 📁 Project Directory Structure

```text
Task-3-Image-Captioning/
├── .venv/                     # Isolated Python virtual environment (ignored by Git)
├── benchmark_samples/         # Built-in reference images (Bakery Logo, Landscape, Animal, Person)
│   ├── bakery_logo.png
│   ├── landscape_sample.jpg
│   ├── animal_sample.jpg
│   └── person_sample.jpg
├── model.py                   # Hybrid multimodal inference engine & caption synthesizer
├── app.py                     # Streamlit web interface with mode selector & OCR cards
├── ocr_engine.py              # Native Windows OCR integration with text cleaning
├── image_analysis.py          # Deterministic color palette & composition analyzer
├── create_test_images.py      # Script generating reproducible benchmark test images
├── run_benchmarks.py          # Automated benchmark runner across all test images
├── test_captioning.py         # Automated unit test suite (9 tests)
├── smoke_test.py              # End-to-end CPU smoke verification script
├── requirements.txt           # CPU-optimized dependency specifications
└── README.md                  # Comprehensive documentation
```

---

## 🚀 Setup & Execution (Windows)

### 1. Virtual Environment Setup

```powershell
cd D:\CODSOFT_TASKSNO\Task-3-Image-Captioning
py -m venv .venv
.\.venv\Scripts\activate
```

### 2. Configure Hugging Face Cache (Drive D)

Point `HF_HOME` to Drive D to preserve system drive storage:

```powershell
$env:HF_HOME = "D:\AI_Model_Cache\huggingface"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

### 3. Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 🌐 Running the Streamlit Web Application

```powershell
$env:HF_HOME = "D:\AI_Model_Cache\huggingface"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Benchmark & Quality Verification

Tested across four distinct image categories:
1. **Bakery Logo (SupriLicious Cakes):** Text, slogan, circular badge, pink background, balloons, bunting.
2. **Scenic Landscape Vista:** Mountains, sunrise sky, lake.
3. **Animal in Nature:** Golden dog standing in green grass under open sky.
4. **Person at Workplace:** Person seated at wooden desk working on laptop.

### Benchmark Metrics (Tested on Windows Laptop with 8 GB RAM):
- **Model Load Time (Cached on Drive D):** **`2.46 seconds`**
- **Peak Process RAM Usage:** **`1,431.08 MB (~1.40 GB)`**
- **Average CPU Inference Latency:** **`6.5s` (Short)** to **`12.5s` (Detailed)**
- **Unit Tests:** **9 / 9 passed** in `0.44s`.

Run the automated test suite:
```powershell
.\.venv\Scripts\python.exe -m unittest test_captioning.py -v
```

Run the full benchmark runner:
```powershell
$env:HF_HOME = "D:\AI_Model_Cache\huggingface"
.\.venv\Scripts\python.exe run_benchmarks.py
```

---

## ⚠️ Known Limitations

1. **CPU Latency:** Full multi-pass inference (BLIP visual + background + OCR + color analysis) requires ~6 to 13 seconds on CPU.
2. **Extreme Text Distortion:** Highly stylized calligraphy or circular curved text at low resolutions may be partially clipped by OCR; high-resolution images (>600px) yield the best results.
3. **Hardware Headroom:** Operates comfortably within 8 GB RAM systems with ~1.4 GB peak working set.
