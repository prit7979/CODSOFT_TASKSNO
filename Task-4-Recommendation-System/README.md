# Task 4: Movie Recommendation System

A content-based movie recommendation system built as part of the **CodSoft Artificial Intelligence Internship**. The system suggests movies similar to a given title by analyzing textual metadata using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity**.

---

## Dataset & Attribution

- **Dataset**: [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (MovieLens & TMDB Metadata)
- **Source**: Kaggle `movies_metadata.csv` (45,466 records) merged with `keywords.csv` (46,419 records).
- **Attribution**: This product uses data sourced from [The Movie Database (TMDB)](https://www.themoviedb.org/) and [GroupLens Research](https://grouplens.org/datasets/movielens/), but is not endorsed, certified, or affiliated with TMDB or GroupLens.
- **Cleaning & Filtering Pipeline**:
  - Filtered corrupted non-numeric IDs and missing titles.
  - Merged keywords via numeric movie IDs.
  - Parsed JSON-like metadata lists into clean strings.
  - Deduplicated movie IDs and titles (prioritizing highest vote counts).
  - Retained **42,227** valid, unique movie records.
- **Local Storage**: Processed clean dataset stored at `data/movies.csv` (~16.8 MB), retaining only `title`, `genres`, `keywords`, and `overview`.

---

## Architecture & Memory Efficiency (8 GB RAM Optimization)

Standard recommendation tutorials compute an $N \times N$ dense similarity matrix, which for 42,227 movies would consume **~15.4 GB of RAM**, crashing an 8 GB laptop.

This system is engineered for low-memory environments:
1. **Feature Soup Construction**: Combines genres, keywords, and plot overviews into a single normalized text representation per movie.
2. **Sparse TF-IDF Representation**: Vectorizes text into a memory-efficient SciPy CSR sparse matrix (~10 MB RAM).
3. **On-the-Fly Cosine Similarity**: Never computes or stores a dense $N \times N$ matrix. When a user requests recommendations, cosine similarity is computed only for that single movie vector against the sparse catalog matrix ($1 \times 42,227$ evaluation in ~45 ms, consuming < 50 MB total RAM).
4. **Ranking & Filtering**: Uses `np.argpartition` to extract top candidates, excludes self-matches, and returns the top-$k$ nearest neighbors.

---

## Project Structure

```text
Task-4-Recommendation-System/
├── app.py                 # Streamlit web interface
├── recommender.py         # Recommendation engine & dataset loading module
├── test_recommender.py    # Unit tests for engine & loader
├── requirements.txt       # Python dependencies
├── data/
│   └── movies.csv         # Cleaned TMDB 5000 dataset (~2 MB)
└── README.md              # Project documentation & dataset attribution
```

---

## Installation & Running

### 1. Activate Environment
```bash
# Windows
.venv\Scripts\activate

# Unix / macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Unit Tests
```bash
python -m unittest test_recommender.py -v
```

### 4. Launch Streamlit Web App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
