# CodSoft Artificial Intelligence Internship Tasks

This repository documents the five projects completed as part of the **CodSoft Artificial Intelligence Internship**. The repository encompasses practical implementations across natural language processing, game theory and adversarial search, multimodal vision-language captioning, content-based recommendation systems, and deep learning computer vision.

All tasks are engineered to run reliably on Windows systems with CPU-only inference and standard memory constraints (8 GB RAM).

---

## 📚 Projects Overview

### 1. Task 1 — Rule-Based Chatbot
- **Directory:** [`Task-1-Rule-Based-Chatbot/`](Task-1-Rule-Based-Chatbot/README.md)
- **Objective:** Build a deterministic, lightweight conversational chatbot that responds to user inputs using predefined rules and keyword matching.
- **Approach:**
  - Employs an input normalization pipeline that strips trailing punctuation, collapses whitespace, and converts tokens to lowercase.
  - Implements prioritized intent matching via regular expressions (`re`) and keyword collections across diverse categories (greetings, identity inquiries, help commands, real-time date/time, programming tips, humor, and pleasantries).
  - Selects varied responses dynamically from curated response pools using `random.choice`.
  - Built strictly with zero external dependencies (Python standard library only).
- **Interface:** Interactive terminal-based conversational loop.
- **Documentation:** [View Task 1 README](Task-1-Rule-Based-Chatbot/README.md)

---

### 2. Task 2 — Tic-Tac-Toe AI
- **Directory:** [`Task-2-Tic-Tac-Toe-AI/`](Task-2-Tic-Tac-Toe-AI/README.md)
- **Objective:** Develop an optimal, unbeatable AI agent to play Tic-Tac-Toe against a human player.
- **Approach:**
  - Implements the **Minimax** adversarial decision-making algorithm with recursive backtracking.
  - Features **Alpha-Beta Pruning** ($\alpha$-$\beta$ cutoffs) and strategic move ordering (Center $\rightarrow$ Corners $\rightarrow$ Edges) to minimize explored tree branches and deliver instant responses.
  - Applies depth-based scoring penalties (`+10 - depth` for AI wins, `-10 + depth` for human wins) to incentivize rapid victories and resilient defense.
  - Mathematically guarantees that the AI cannot be defeated (results in either an AI win or a draw).
  - Built strictly with zero external dependencies (Python standard library only).
- **Interface:** Interactive terminal-based game loop with an intuitive 1–9 coordinate board guide and marker selection (`X` or `O`).
- **Documentation:** [View Task 2 README](Task-2-Tic-Tac-Toe-AI/README.md)

---

### 3. Task 3 — AI Image Captioning
- **Directory:** [`Task-3-Image-Captioning/`](Task-3-Image-Captioning/README.md)
- **Objective:** Generate specific, detailed, and accurate natural language descriptions of images rather than short generic phrases.
- **Approach:**
  - Deploys **Salesforce BLIP** (`Salesforce/blip-image-captioning-base`, ~224M parameters) for vision-language semantic inference running 100% locally on CPU.
  - Integrates **multi-region semantic probing** to analyze both global scene semantics and localized focal motifs (e.g. illustrations, cupcakes, people, animals).
  - Integrates **Native Windows Media OCR** (`winocr` / `Windows.Media.Ocr`) for zero-overhead, accurate extraction of brand names, slogans, and typography without cloud APIs.
  - Employs computational color and layout analysis (`image_analysis.py`) to classify background tones, dominant RGB palettes, circular badge geometry, and decorative accents.
  - Supports three distinct caption modes: **Short** (concise headline), **Detailed** (fluent 2–3 sentence narrative), and **Very Detailed** (structured multi-section breakdown).
- **Interface:** Interactive Streamlit web application with image upload, built-in benchmark samples, visual metadata display, and `.txt` caption downloads.
- **Documentation:** [View Task 3 README](Task-3-Image-Captioning/README.md)

---

### 4. Task 4 — Movie Recommendation System
- **Directory:** [`Task-4-Recommendation-System/`](Task-4-Recommendation-System/README.md)
- **Objective:** Build a content-based recommendation engine that suggests movies similar to a user's selection based on metadata attributes.
- **Approach:**
  - Merges and normalizes movie metadata (genres, keywords, and plot overviews) into a cohesive textual feature soup.
  - Constructs a sparse **TF-IDF (Term Frequency-Inverse Document Frequency)** matrix representation using `scikit-learn` and `scipy.sparse`.
  - Implements an **on-the-fly Cosine Similarity** evaluation mechanism ($1 \times N$ vector computation) that avoids materializing dense $N \times N$ matrices, operating comfortably under 50 MB of memory on an 8 GB RAM laptop.
  - Evaluated on a cleaned dataset of 42,227 films sourced from [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (TMDB & MovieLens).
- **Interface:** Interactive Streamlit web dashboard with instant autocomplete search, similarity score displays, and metadata inspection.
- **Documentation:** [View Task 4 README](Task-4-Recommendation-System/README.md)

---

### 5. Task 5 — Face Detection & Recognition
- **Directory:** [`Task-5-Face-Detection/`](Task-5-Face-Detection/README.md)
- **Objective:** Accurately localize and count human faces in digital images and real-time webcam captures.
- **Approach:**
  - Leverages OpenCV's deep learning module (`cv2.FaceDetectorYN`) with the lightweight **YuNet ONNX** convolutional neural network (`face_detection_yunet_2023mar.onnx`, 232 KB).
  - Employs dynamic scale normalization for high-resolution images ($>800\text{ px}$) to preserve inference sensitivity before projecting coordinates back to the original dimensions.
  - Features configurable confidence score filtering (`scoreThreshold`) and Non-Maximum Suppression (`nmsThreshold`) to eliminate background false positives and duplicate bounding boxes.
  - Provides visual annotations with customizable bounding-box palettes, index tags, and face counts.
- **Interface:** Interactive Streamlit web application supporting both file uploads (JPG, PNG) and live browser webcam capture (`st.camera_input`) with side-by-side visual comparisons and direct image downloads.
- **Documentation:** [View Task 5 README](Task-5-Face-Detection/README.md)

---

## 🛠️ Technology Stack Summary

| Technology / Library | Used In | Purpose |
| :--- | :--- | :--- |
| **Python 3.13.1** | Tasks 1, 2, 3, 4, 5 | Core programming language across all projects |
| **Python Standard Library** (`re`, `math`, `random`, `unittest`, `datetime`) | Tasks 1, 2 | Pure standard-library implementations for chatbot rules and Minimax search |
| **PyTorch (CPU)** (`torch`, `torchvision`) | Task 3 | Neural network execution on CPU for BLIP vision-language models |
| **Hugging Face Transformers** | Task 3 | BLIP processor and conditional generation (`BlipForConditionalGeneration`) |
| **Windows Media OCR** (`winocr`) | Task 3 | Native Windows 10/11 operating system OCR for zero-overhead text extraction |
| **Pillow** (`PIL`) | Tasks 3, 4, 5 | Image loading, resizing, EXIF normalization, and programmatic graphics generation |
| **OpenCV** (`opencv-python`) | Task 5 | Computer vision operations and deep learning inference via `cv2.FaceDetectorYN` |
| **YuNet ONNX** | Task 5 | Pre-trained lightweight deep learning face detection model |
| **Scikit-Learn** & **SciPy** | Task 4 | TF-IDF text vectorization, sparse matrix operations, and cosine similarity ranking |
| **NumPy** & **Pandas** | Tasks 4, 5 | Matrix array manipulations, dataset parsing, and bounding box computations |
| **Streamlit** | Tasks 3, 4, 5 | Modern web dashboard interfaces for browser-based user interaction |

---

## 📁 Repository Structure

```text
CODSOFT_TASKSNO/
├── .gitignore
├── README.md                          # Root repository documentation (this file)
├── Task-1-Rule-Based-Chatbot/
│   ├── chatbot.py                     # Rule-based conversational logic & CLI
│   ├── test_chatbot.py                # Unit test suite
│   ├── requirements.txt               # Dependencies (standard library only)
│   └── README.md                      # Task 1 documentation
├── Task-2-Tic-Tac-Toe-AI/
│   ├── game.py                        # Board state & win/draw logic
│   ├── ai.py                          # Minimax algorithm with Alpha-Beta pruning
│   ├── main.py                        # Interactive CLI game loop
│   ├── test_game.py                   # Unit test suite (21 tests)
│   ├── requirements.txt               # Dependencies (standard library only)
│   └── README.md                      # Task 2 documentation
├── Task-3-Image-Captioning/
│   ├── app.py                         # Streamlit web dashboard
│   ├── model.py                       # BLIP inference & multi-region caption synthesizer
│   ├── ocr_engine.py                  # Native Windows Media OCR integration
│   ├── image_analysis.py              # Color palette & geometric composition analyzer
│   ├── test_captioning.py             # Unit test suite (9 tests)
│   ├── run_benchmarks.py              # Benchmark runner across diverse samples
│   ├── requirements.txt               # Dependencies (torch-cpu, transformers, winocr, streamlit)
│   └── README.md                      # Task 3 documentation
├── Task-4-Recommendation-System/
│   ├── app.py                         # Streamlit recommendation interface
│   ├── recommender.py                 # TF-IDF & on-the-fly cosine similarity engine
│   ├── test_recommender.py            # Unit test suite
│   ├── requirements.txt               # Dependencies (scikit-learn, pandas, streamlit)
│   └── README.md                      # Task 4 documentation
└── Task-5-Face-Detection/
    ├── app.py                         # Streamlit face detection web application
    ├── detector.py                    # OpenCV YuNet FaceDetector wrapper
    ├── test_detector.py               # Unit test suite (22 tests)
    ├── requirements.txt               # Dependencies (opencv-python, numpy, streamlit)
    └── README.md                      # Task 5 documentation
```

---

## 🚀 Environment & Setup Guidelines

Each task is self-contained with its own isolated virtual environment (`.venv`) and dependency specifications.

### Running Any Task:

1. **Navigate to the target task folder:**
   ```powershell
   cd D:\CODSOFT_TASKSNO\Task-X-...
   ```

2. **Activate the task's virtual environment:**
   ```powershell
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   ```

3. **Execute the project:**
   - **Console applications (Tasks 1 & 2):**
     ```powershell
     .\.venv\Scripts\python.exe chatbot.py   # Task 1
     .\.venv\Scripts\python.exe main.py      # Task 2
     ```
   - **Streamlit web dashboards (Tasks 3, 4 & 5):**
     ```powershell
     .\.venv\Scripts\python.exe -m streamlit run app.py
     ```

4. **Run automated unit tests:**
   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -v
   ```
