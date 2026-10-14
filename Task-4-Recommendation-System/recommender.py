"""
Content-Based Movie Recommendation Engine.

This module implements a memory-efficient content-based recommendation system
using TF-IDF (Term Frequency-Inverse Document Frequency) and on-the-fly Cosine
Similarity. Designed specifically to run on consumer hardware (such as 8 GB RAM
laptops) without precomputing or storing a massive N x N dense similarity matrix.
"""

import os
import ast
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "movies.csv")


def _parse_tmdb_json_field(val: Any) -> str:
    """
    Safely parse TMDB JSON string lists (e.g., '[{"name": "Action"}]')
    into human-readable comma-separated strings.
    """
    if pd.isna(val) or val is None:
        return ""
    val_str = str(val).strip()
    if not val_str or val_str == "[]":
        return ""
    if val_str.startswith("[") and ("name" in val_str):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                names = [
                    item["name"]
                    for item in parsed
                    if isinstance(item, dict) and "name" in item
                ]
                return ", ".join(names)
        except Exception:
            pass
    return val_str


def clean_and_prepare_dataset(
    movies_metadata_path: str,
    keywords_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Merges movies_metadata.csv with keywords.csv, cleans invalid IDs,
    filters missing/duplicate titles, normalizes metadata, and saves
    the resulting dataset.

    Args:
        movies_metadata_path: Path to movies_metadata.csv.
        keywords_path: Path to keywords.csv.
        output_path: Optional destination CSV path (e.g. data/movies.csv).

    Returns:
        Cleaned DataFrame with 'title', 'genres', 'keywords', and 'overview'.
    """
    # 1. Read movies_metadata
    mm = pd.read_csv(movies_metadata_path, low_memory=False)

    # Coerce and clean numeric movie IDs
    mm["id_num"] = pd.to_numeric(mm["id"], errors="coerce")
    mm = mm[mm["id_num"].notna()].copy()
    mm["id_num"] = mm["id_num"].astype(int)

    # Clean titles
    mm = mm[mm["title"].notna()].copy()
    mm["title"] = mm["title"].astype(str).str.strip()
    mm = mm[mm["title"] != ""]

    # Sort by vote_count descending if present so deduplication preserves notable entries
    if "vote_count" in mm.columns:
        mm["vote_count_num"] = pd.to_numeric(mm["vote_count"], errors="coerce").fillna(0).astype(int)
        mm = mm.sort_values(by="vote_count_num", ascending=False)
    else:
        mm["vote_count_num"] = 0

    if "vote_average" in mm.columns:
        mm["vote_average_num"] = pd.to_numeric(mm["vote_average"], errors="coerce").fillna(0.0)
    else:
        mm["vote_average_num"] = 0.0

    # Deduplicate movie IDs
    mm = mm.drop_duplicates(subset=["id_num"], keep="first")

    # 2. Read and clean keywords
    kw = pd.read_csv(keywords_path)
    kw["id_num"] = pd.to_numeric(kw["id"], errors="coerce")
    kw = kw[kw["id_num"].notna()].copy()
    kw["id_num"] = kw["id_num"].astype(int)
    kw = kw.drop_duplicates(subset=["id_num"], keep="first")

    # 3. Merge metadata with keywords
    merged = mm.merge(kw[["id_num", "keywords"]], on="id_num", how="left")

    # Deduplicate case-insensitively by title
    merged["title_norm"] = merged["title"].str.lower()
    merged = merged.drop_duplicates(subset=["title_norm"], keep="first").reset_index(drop=True)

    # Parse JSON fields
    merged["genres"] = merged["genres"].apply(_parse_tmdb_json_field)
    merged["keywords"] = merged["keywords"].apply(_parse_tmdb_json_field)
    merged["overview"] = merged["overview"].fillna("").astype(str).str.strip()
    merged["vote_average"] = merged["vote_average_num"].round(1)
    merged["vote_count"] = merged["vote_count_num"]

    final_df = merged[["title", "genres", "keywords", "overview", "vote_average", "vote_count"]].copy()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        final_df.to_csv(output_path, index=False)

    return final_df


def load_movie_dataset(
    filepath: Optional[str] = None, fallback_on_missing: bool = False
) -> pd.DataFrame:
    """
    Loads and normalizes the movie dataset.

    Args:
        filepath: Path to the CSV file. If None, uses DEFAULT_DATASET_PATH.
        fallback_on_missing: If True, returns a minimal fallback dataset if the file is missing.

    Returns:
        A cleaned pandas DataFrame with 'title', 'genres', 'keywords', 'overview',
        and optional 'vote_average', 'vote_count'.

    Raises:
        FileNotFoundError: If the file is not found and fallback_on_missing is False.
        ValueError: If the dataset is empty or lacks the required 'title' column.
    """
    target_path = filepath or DEFAULT_DATASET_PATH

    if not os.path.exists(target_path):
        if fallback_on_missing:
            return get_fallback_dataset()
        raise FileNotFoundError(
            f"Dataset file not found at: '{target_path}'. "
            "Please ensure 'movies.csv' exists or specify a valid filepath."
        )

    df = pd.read_csv(target_path)

    if df.empty:
        raise ValueError(f"Dataset at '{target_path}' is empty.")

    if "title" not in df.columns:
        raise ValueError(f"Dataset at '{target_path}' is missing the required 'title' column.")

    # Ensure required columns exist
    for col in ["genres", "keywords", "overview"]:
        if col not in df.columns:
            df[col] = ""

    # Clean titles and remove duplicates
    df = df.dropna(subset=["title"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"] != ""]
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

    # Normalize fields and parse JSON if present
    df["genres"] = df["genres"].apply(_parse_tmdb_json_field)
    df["keywords"] = df["keywords"].apply(_parse_tmdb_json_field)
    df["overview"] = df["overview"].fillna("").astype(str)

    ret_cols = ["title", "genres", "keywords", "overview"]
    if "vote_average" in df.columns:
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
        ret_cols.append("vote_average")
    if "vote_count" in df.columns:
        df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
        ret_cols.append("vote_count")

    return df[ret_cols]


def get_fallback_dataset() -> pd.DataFrame:
    """
    Returns a small, resilient fallback dataset for testing and offline environments.
    """
    return pd.DataFrame([
        {
            "title": "The Matrix",
            "genres": "Action, Science Fiction",
            "keywords": "virtual reality, simulation, cyberpunk, hacker",
            "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality.",
            "vote_average": 8.2,
            "vote_count": 9000,
        },
        {
            "title": "The Matrix Reloaded",
            "genres": "Action, Science Fiction",
            "keywords": "virtual reality, simulation, machine war, zion, neo",
            "overview": "Freedom fighters struggle against the machine army while Neo seeks answers.",
            "vote_average": 7.0,
            "vote_count": 4500,
        },
        {
            "title": "Inception",
            "genres": "Action, Science Fiction, Thriller",
            "keywords": "dreams, subconscious, heist, reality, mind",
            "overview": "A thief who steals corporate secrets through dream-sharing technology is given an inverse task.",
            "vote_average": 8.3,
            "vote_count": 14000,
        },
        {
            "title": "The Notebook",
            "genres": "Romance, Drama",
            "keywords": "love, relationship, summer, diary, memory",
            "overview": "A poor yet passionate young man falls in love with a rich young woman.",
            "vote_average": 7.8,
            "vote_count": 3500,
        },
        {
            "title": "Toy Story",
            "genres": "Animation, Adventure, Comedy",
            "keywords": "toys, friendship, childhood, woody, buzz",
            "overview": "A cowboy doll is threatened when a new spaceman figure supplants him.",
            "vote_average": 7.9,
            "vote_count": 5500,
        }
    ])


def extract_unique_genres(df: pd.DataFrame) -> List[str]:
    """
    Extracts all unique genre labels from the movie dataframe.
    """
    genres_set = set()
    if "genres" in df.columns:
        for g_str in df["genres"].dropna():
            for g in str(g_str).split(","):
                cleaned = g.strip()
                if cleaned:
                    genres_set.add(cleaned)
    return sorted(list(genres_set))


def explore_by_category(
    df: pd.DataFrame,
    selected_genres: List[str],
    min_rating: float = 0.0,
    top_k: int = 10,
    min_vote_count: int = 20,
) -> List[Dict[str, Any]]:
    """
    Explore and rank movies matching one or more selected genres.

    Ranking Criteria:
    1. Genre overlap score (how many of the selected genres match).
    2. Reliable rating / popularity (vote_average and vote_count).

    Args:
        df: Movie DataFrame with 'title', 'genres', 'overview', and optional 'vote_average', 'vote_count'.
        selected_genres: List of selected genre names.
        min_rating: Optional minimum rating filter (default: 0.0).
        top_k: Number of recommendations to return (default: 10).
        min_vote_count: Minimum votes required when min_rating > 0.0 (default: 20).

    Returns:
        List of movie dictionaries with title, genres, rating, overview, and match statistics.
    """
    if df.empty or not selected_genres:
        return []

    target_genres = {g.strip().lower() for g in selected_genres if g.strip()}
    if not target_genres:
        return []

    candidates = []
    seen_titles = set()

    has_ratings = "vote_average" in df.columns
    has_votes = "vote_count" in df.columns

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title or title.lower() in seen_titles:
            continue

        g_str = str(row.get("genres", ""))
        movie_genres = [g.strip() for g in g_str.split(",") if g.strip()]
        matched = [g for g in movie_genres if g.lower() in target_genres]
        if not matched:
            continue

        rating = float(row["vote_average"]) if has_ratings and pd.notna(row.get("vote_average")) else None
        votes = int(row["vote_count"]) if has_votes and pd.notna(row.get("vote_count")) else 0

        # Apply minimum rating filter if specified and reliable
        if min_rating > 0.0 and rating is not None:
            if rating < min_rating or votes < min_vote_count:
                continue

        seen_titles.add(title.lower())
        candidates.append({
            "title": title,
            "genres": row.get("genres", ""),
            "rating": rating,
            "vote_count": votes,
            "overview": row.get("overview", ""),
            "match_count": len(matched),
            "matched_genres": matched,
        })

    # Sort primarily by number of matched genres, then rating, then vote count
    candidates.sort(
        key=lambda x: (
            -x["match_count"],
            -(x["rating"] if x["rating"] is not None else 0.0),
            -x["vote_count"]
        )
    )

    return candidates[:top_k]


def get_movie_suggestions(
    query: str, titles: List[str], max_suggestions: int = 8
) -> List[str]:
    """
    Returns up to max_suggestions matching titles for a given query.
    Case-insensitive, strips extra whitespace.
    Prioritizes prefix matches, followed by word-initial matches, then substring matches.

    Args:
        query: User typed query string.
        titles: List of candidate movie titles.
        max_suggestions: Maximum number of suggestions to return (default: 8).

    Returns:
        List of matching movie titles up to max_suggestions.
    """
    clean_q = query.strip().lower()
    if not clean_q:
        return []

    prefix_matches = []
    word_matches = []
    contains_matches = []

    for t in titles:
        tl = t.lower()
        if tl.startswith(clean_q):
            prefix_matches.append(t)
        elif any(w.startswith(clean_q) for w in tl.replace("-", " ").split()):
            word_matches.append(t)
        elif clean_q in tl:
            contains_matches.append(t)

    combined = prefix_matches + word_matches + contains_matches
    seen = set()
    result = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            result.append(item)
        if len(result) >= max_suggestions:
            break

    return result


class MovieRecommender:
    """
    Content-Based Movie Recommender using TF-IDF and on-the-fly Cosine Similarity.
    Memory-efficient: avoids storing an N x N dense similarity matrix.
    """

    def __init__(self, stop_words: str = "english", max_features: Optional[int] = 35000):
        """
        Initialize the recommender.

        Args:
            stop_words: Language stop words to filter out during vectorization.
            max_features: Maximum vocabulary features to bound memory consumption.
        """
        self.stop_words = stop_words
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            max_features=self.max_features
        )
        self.df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.cosine_sim = None  # Intentionally kept None to avoid N x N dense RAM allocation
        self.title_to_index: Dict[str, int] = {}

    @staticmethod
    def _clean_text(value: Any) -> str:
        """
        Convert any metadata value into clean, normalized text.
        Handles None, NaN, numeric types, and string padding.
        """
        if pd.isna(value) or value is None:
            return ""
        return str(value).strip().lower()

    def build_feature_soup(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine genres, keywords, and description/overview into a single text representation
        ("metadata soup") for each movie.

        Args:
            df: DataFrame containing movie records.

        Returns:
            A pandas Series of combined text strings.
        """
        soup_series = []
        for _, row in df.iterrows():
            parts = []

            # Extract genres if present
            if "genres" in row:
                genre_text = self._clean_text(row["genres"])
                if genre_text:
                    parts.append(genre_text)

            # Extract keywords if present
            if "keywords" in row:
                keyword_text = self._clean_text(row["keywords"])
                if keyword_text:
                    parts.append(keyword_text)

            # Extract description / overview if present
            overview_col = None
            if "overview" in row:
                overview_col = "overview"
            elif "description" in row:
                overview_col = "description"

            if overview_col:
                desc_text = self._clean_text(row[overview_col])
                if desc_text:
                    parts.append(desc_text)

            # Combine parts or fallback to clean title if all metadata is missing
            soup = " ".join(parts).strip()
            if not soup and "title" in row:
                soup = self._clean_text(row["title"])
            soup_series.append(soup)

        return pd.Series(soup_series, index=df.index)

    def fit(self, df: pd.DataFrame) -> "MovieRecommender":
        """
        Prepares movie data, handles missing values/duplicates,
        and computes the sparse TF-IDF feature matrix.

        NOTE: Does NOT precompute an N x N dense similarity matrix.
        Similarity is computed on-demand for individual movie queries.

        Args:
            df: DataFrame with at least a 'title' column, along with optional
                'genres', 'keywords', and 'overview'/'description' columns.

        Returns:
            Self (fitted recommender).
        """
        if df.empty:
            raise ValueError("Cannot fit recommender on an empty DataFrame.")
        if "title" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'title' column.")

        # Create a working copy and drop rows with empty/null titles
        data = df.copy()
        data = data[data["title"].notna() & (data["title"].astype(str).str.strip() != "")]

        # Deduplicate titles (keep the first occurrence)
        data = data.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

        # Store cleaned metadata
        self.df = data

        # Build feature soup and compute sparse TF-IDF matrix
        soup = self.build_feature_soup(self.df)
        self.tfidf_matrix = self.vectorizer.fit_transform(soup)

        # Do NOT compute or store N x N dense cosine_sim matrix to conserve RAM on 8 GB systems
        self.cosine_sim = None

        # Build case-insensitive title lookup mapping
        self.title_to_index = {
            title.strip().lower(): idx for idx, title in enumerate(self.df["title"])
        }

        return self

    def recommend(self, title: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find and return the top_k most similar movies for a given movie title.
        Calculates cosine similarity on-the-fly for only the target movie vector.

        Args:
            title: The title of the movie to query.
            top_k: Number of recommendations to return (default: 5).

        Returns:
            A list of dictionaries with recommended movie metadata and similarity scores:
            [
                {
                    "title": str,
                    "similarity_score": float,
                    "genres": Any,
                    "overview": Any
                },
                ...
            ]

        Raises:
            RuntimeError: If called before calling `fit`.
            ValueError: If the title is not found in the dataset.
        """
        if self.df is None or self.tfidf_matrix is None:
            raise RuntimeError("Recommender is not fitted. Please call `fit` first.")

        if not title or not isinstance(title, str):
            raise ValueError("Movie title must be a non-empty string.")

        normalized_title = title.strip().lower()
        if normalized_title not in self.title_to_index:
            raise ValueError(f"Movie '{title}' was not found in the dataset.")

        movie_idx = self.title_to_index[normalized_title]

        # Calculate cosine similarity ONLY for the target movie vector (1 x N)
        target_vector = self.tfidf_matrix[movie_idx]
        sim_scores = cosine_similarity(target_vector, self.tfidf_matrix).ravel()

        # Retrieve top candidates efficiently (excluding self)
        num_candidates = min(top_k + 1, len(sim_scores))
        if num_candidates <= 1:
            return []

        # Use argpartition to find top scores in O(N) without full array sort
        candidate_indices = np.argpartition(sim_scores, -num_candidates)[-num_candidates:]
        # Sort only those top candidates descending
        sorted_top_indices = candidate_indices[np.argsort(-sim_scores[candidate_indices])]

        recommendations: List[Dict[str, Any]] = []
        for idx in sorted_top_indices:
            if idx == movie_idx:
                continue  # Exclude the queried movie itself

            row = self.df.iloc[idx]
            rec_item = {
                "title": row["title"],
                "similarity_score": round(float(sim_scores[idx]), 4),
            }
            if "genres" in row and pd.notna(row["genres"]) and str(row["genres"]).strip() != "":
                rec_item["genres"] = row["genres"]
            if "overview" in row and pd.notna(row["overview"]) and str(row["overview"]).strip() != "":
                rec_item["overview"] = row["overview"]
            elif "description" in row and pd.notna(row["description"]) and str(row["description"]).strip() != "":
                rec_item["overview"] = row["description"]

            recommendations.append(rec_item)
            if len(recommendations) >= top_k:
                break

        return recommendations


def get_movie_recommendations(
    df: pd.DataFrame, title: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience functional interface to fit and retrieve recommendations directly.

    Args:
        df: Movie metadata DataFrame.
        title: Title of the movie to search for.
        top_k: Number of recommendations to return (default: 5).

    Returns:
        List of recommended movie dictionaries.
    """
    recommender = MovieRecommender()
    recommender.fit(df)
    return recommender.recommend(title, top_k=top_k)
