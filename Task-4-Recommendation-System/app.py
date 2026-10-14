"""
CineMatch AI - Intelligent Movie Discovery Application.
Content-based recommendation engine powered by TF-IDF and Cosine Similarity,
paired with multi-genre Category Exploration and rating filters.
"""

from typing import List, Dict, Any, Tuple
import streamlit as st
import pandas as pd
from recommender import (
    MovieRecommender,
    load_movie_dataset,
    get_fallback_dataset,
    get_movie_suggestions,
    extract_unique_genres,
    explore_by_category,
)


# Application configuration
st.set_page_config(
    page_title="CineMatch AI - Discover Your Next Favourite Film",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Premium Cinematic Design System with Original Native Font
THEME_CSS = """
<style>
/* Base surfaces using original Streamlit native font stack */
.stApp {
    background-color: #0B0D14 !important;
    color: #F4F1EA !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #121622 !important;
    border-right: 1px solid #292F40 !important;
    min-width: 320px;
}

[data-testid="stSidebar"] * {
    color: #F4F1EA;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #A3A9B8 !important;
}

/* Typography Hierarchy */
.app-brand-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #F4F1EA;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}

.app-brand-subtitle {
    font-size: 0.98rem;
    color: #A3A9B8;
    margin-top: 4px;
    margin-bottom: 24px;
}

.sidebar-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F4F1EA;
    letter-spacing: -0.01em;
    margin-bottom: 2px;
}

.sidebar-subtitle {
    font-size: 0.85rem;
    color: #A3A9B8;
    margin-bottom: 20px;
    line-height: 1.4;
}

.section-heading {
    font-size: 1.25rem;
    font-weight: 600;
    color: #F4F1EA;
    letter-spacing: -0.01em;
    margin-top: 8px;
    margin-bottom: 16px;
}

.control-label {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #A3A9B8;
    margin-bottom: 6px;
    display: block;
}

/* Interactive form inputs */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #191F2E !important;
    border: 1px solid #292F40 !important;
    border-radius: 8px !important;
    color: #F4F1EA !important;
    font-size: 0.95rem !important;
    min-height: 44px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: #8B7CFF !important;
    box-shadow: 0 0 0 2px rgba(139, 124, 255, 0.25) !important;
}

/* Prevent Streamlit 'Press Enter to apply' badge from overlapping text inside input fields */
[data-testid="InputInstructions"] {
    display: none !important;
}

/* Elevated Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #191F2E !important;
    border: 1px solid #292F40 !important;
    border-radius: 10px !important;
    padding: 1.2rem !important;
}

/* Selected Movie Hero Banner */
.hero-movie-card {
    background-color: #191F2E;
    border: 1px solid #292F40;
    border-left: 3px solid #8B7CFF;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 28px;
}

.hero-movie-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #F4F1EA;
    margin: 0 0 8px 0;
}

.hero-movie-overview {
    font-size: 0.92rem;
    color: #A3A9B8;
    line-height: 1.55;
    margin: 8px 0 0 0;
}

/* Movie Card Components */
.movie-title-text {
    font-size: 1.05rem;
    font-weight: 600;
    color: #F4F1EA;
    margin: 0 0 8px 0;
    line-height: 1.35;
    height: 2.7em;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.badge-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}

.rating-pill {
    display: inline-flex;
    align-items: center;
    background: rgba(245, 199, 107, 0.12);
    border: 1px solid rgba(245, 199, 107, 0.3);
    color: #F5C76B;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
}

.similarity-pill {
    display: inline-flex;
    align-items: center;
    background: rgba(139, 124, 255, 0.12);
    border: 1px solid rgba(139, 124, 255, 0.3);
    color: #8B7CFF;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.78rem;
}

.match-count-text {
    font-size: 0.78rem;
    color: #69C7E8;
    font-weight: 500;
}

.vote-count-text {
    font-size: 0.78rem;
    color: #A3A9B8;
}

.genre-tag-list {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 10px;
}

.genre-tag-item {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid #292F40;
    color: #A3A9B8;
    font-size: 0.74rem;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 4px;
}

.card-overview-text {
    font-size: 0.84rem;
    color: #A3A9B8;
    line-height: 1.5;
    margin-top: 6px;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    height: 5.1em;
}

/* Progress bar accent */
[data-testid="stProgress"] > div > div > div > div {
    background-color: #8B7CFF !important;
}

/* Multiselect tags */
[data-baseweb="tag"] {
    background-color: #191F2E !important;
    border: 1px solid #292F40 !important;
    border-radius: 6px !important;
}

[data-baseweb="tag"] span {
    color: #F4F1EA !important;
}

[data-baseweb="tag"] svg {
    fill: #A3A9B8 !important;
}

/* Slider thumb & track accents */
[data-testid="stSlider"] div[role="slider"] {
    background-color: #8B7CFF !important;
    border-color: #8B7CFF !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"] div div {
    background-color: #8B7CFF !important;
}

/* Radio button active dot */
[data-testid="stRadio"] label span[data-checked="true"],
[data-testid="stRadio"] div[role="radiogroup"] label div[data-checked="true"] {
    background-color: #8B7CFF !important;
    border-color: #8B7CFF !important;
}

/* Subtle Divider */
hr {
    border-color: #292F40 !important;
    margin: 1.2rem 0 !important;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_data
def get_dataset() -> Tuple[pd.DataFrame, bool]:
    """
    Loads movie dataset from local storage.
    Returns (dataframe, is_fallback).
    """
    try:
        data = load_movie_dataset(fallback_on_missing=False)
        return data, False
    except Exception:
        fallback_data = get_fallback_dataset()
        return fallback_data, True


@st.cache_resource
def load_recommender(data: pd.DataFrame) -> MovieRecommender:
    """Initializes and fits the MovieRecommender instance."""
    rec = MovieRecommender()
    rec.fit(data)
    return rec


def format_genres_html(genres_str: Any) -> str:
    """Format comma-separated genres string into subtle pill tags."""
    if pd.isna(genres_str) or not str(genres_str).strip():
        return '<span class="genre-tag-item">General</span>'
    tags = [g.strip() for g in str(genres_str).split(",") if g.strip()]
    return "".join([f'<span class="genre-tag-item">{tag}</span>' for tag in tags[:4]])


def main():
    movies_df, is_fallback = get_dataset()
    recommender = load_recommender(movies_df)

    # ==========================
    # Sidebar
    # ==========================
    st.sidebar.markdown('<div class="sidebar-title">🎬 CineMatch AI</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-subtitle">Intelligent movie discovery tailored to your preferences.</div>',
        unsafe_allow_html=True
    )

    if is_fallback:
        st.sidebar.warning("⚠️ Using offline fallback dataset. Full catalog unavailable.")

    # Mode Selection with three clear options
    st.sidebar.markdown('<span class="control-label">🔍 Select Mode:</span>', unsafe_allow_html=True)
    mode = st.sidebar.radio(
        "Navigation Mode",
        ["Choose from List", "Manual Search", "Explore by Category"],
        index=0,
        label_visibility="collapsed"
    )

    all_titles_alpha = sorted(movies_df["title"].tolist())
    all_titles_catalog = movies_df["title"].tolist()
    available_genres = extract_unique_genres(movies_df)
    has_ratings = "vote_average" in movies_df.columns

    selected_title = ""
    selected_genres: List[str] = []
    min_rating = 0.0

    # Prominent Interactive Input Card
    with st.sidebar.container(border=True):
        if mode == "Choose from List":
            st.markdown('<span class="control-label">📽️ Select a Movie:</span>', unsafe_allow_html=True)

            default_index = 0
            preferred_defaults = ["The Dark Knight", "Inception", "Avatar", "The Matrix"]
            for p in preferred_defaults:
                if p in all_titles_alpha:
                    default_index = all_titles_alpha.index(p)
                    break

            selected_title = st.selectbox(
                "Select a movie:",
                options=all_titles_alpha,
                index=default_index,
                label_visibility="collapsed"
            )

        elif mode == "Manual Search":
            st.markdown('<span class="control-label">⌨️ Enter Movie Title:</span>', unsafe_allow_html=True)

            query = st.text_input(
                "Search movie title:",
                value="The Dark Knight",
                placeholder="e.g. Spider-Man, Batman...",
                label_visibility="collapsed"
            )

            clean_query = query.strip()
            suggestions = get_movie_suggestions(clean_query, all_titles_catalog, max_suggestions=8)

            if clean_query:
                if suggestions:
                    st.markdown('<span class="control-label" style="margin-top: 10px;">💡 Matching Suggestions:</span>', unsafe_allow_html=True)
                    default_s_idx = 0
                    for i, s in enumerate(suggestions):
                        if s.lower() == clean_query.lower():
                            default_s_idx = i
                            break

                    chosen_suggestion = st.selectbox(
                        "Suggested titles:",
                        options=suggestions,
                        index=default_s_idx,
                        key="manual_suggestion_select",
                        label_visibility="collapsed"
                    )
                    selected_title = chosen_suggestion
                else:
                    st.warning(f"No movie titles found matching '{clean_query}'.")
                    selected_title = clean_query
            else:
                selected_title = ""

        else:
            # Explore by Category Mode
            st.markdown('<span class="control-label">🎭 Select Genre(s):</span>', unsafe_allow_html=True)

            default_genres = [g for g in ["Action", "Science Fiction"] if g in available_genres]
            if not default_genres and available_genres:
                default_genres = [available_genres[0]]

            selected_genres = st.multiselect(
                "Select genres:",
                options=available_genres,
                default=default_genres,
                label_visibility="collapsed"
            )

            if has_ratings:
                st.markdown('<span class="control-label" style="margin-top: 12px;">⭐ Minimum Rating:</span>', unsafe_allow_html=True)
                min_rating = st.slider(
                    "Minimum rating:",
                    min_value=0.0,
                    max_value=9.0,
                    value=6.5,
                    step=0.5,
                    format="%.1f ★",
                    label_visibility="collapsed"
                )

    # Number of Recommendations Slider
    st.sidebar.markdown('<span class="control-label" style="margin-top: 14px;">🔢 Number of Recommendations:</span>', unsafe_allow_html=True)
    top_k = st.sidebar.slider(
        "Recommendations Count:",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
        label_visibility="collapsed"
    )

    # ==========================
    # Main Showcase
    # ==========================
    st.markdown('<h1 class="app-brand-title">🍿 CineMatch AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="app-brand-subtitle">Discover your next favourite film.</div>', unsafe_allow_html=True)

    if mode == "Explore by Category":
        if not selected_genres:
            st.info("👆 Please select at least one genre from the sidebar to explore movies.")
            return

        genres_display = " • ".join(selected_genres)
        rating_info = f" • Rated ≥ {min_rating:.1f}★" if min_rating > 0 else ""
        st.markdown(
            f'<div class="section-heading">✨ Top {top_k} Movies in: {genres_display}{rating_info}</div>',
            unsafe_allow_html=True
        )

        category_results = explore_by_category(
            movies_df,
            selected_genres=selected_genres,
            min_rating=min_rating,
            top_k=top_k
        )

        if not category_results:
            st.warning("No titles found matching the selected criteria.")
            st.caption("💡 Try lowering the minimum rating threshold or choosing different genres.")
            return

        # Display category cards in 3-column rows
        COLS_PER_ROW = 3
        for row_start in range(0, len(category_results), COLS_PER_ROW):
            row_recs = category_results[row_start : row_start + COLS_PER_ROW]
            cols = st.columns(COLS_PER_ROW)
            for idx_in_row, movie in enumerate(row_recs):
                overall_idx = row_start + idx_in_row
                with cols[idx_in_row]:
                    with st.container(border=True):
                        st.markdown(
                            f'<div class="movie-title-text">{overall_idx + 1}. {movie["title"]}</div>',
                            unsafe_allow_html=True
                        )

                        # Badges Row
                        rating_val = movie.get("rating")
                        rating_badge = f'<span class="rating-pill">⭐ {rating_val:.1f} / 10</span>' if rating_val and rating_val > 0 else ""
                        vote_cnt = movie.get("vote_count", 0)
                        vote_text = f'<span class="vote-count-text">({vote_cnt:,})</span>' if vote_cnt > 0 else ""
                        matches_badge = f'<span class="match-count-text">🎯 {movie.get("match_count", 1)}/{len(selected_genres)} match</span>'

                        st.markdown(
                            f'<div class="badge-row">{rating_badge} {vote_text} {matches_badge}</div>',
                            unsafe_allow_html=True
                        )

                        # Genre Tags
                        genre_html = format_genres_html(movie.get("genres", ""))
                        st.markdown(f'<div class="genre-tag-list"><span style="font-size:0.75rem; color:#A3A9B8; margin-right:4px;">🎭</span>{genre_html}</div>', unsafe_allow_html=True)

                        # Synopsis
                        overview = movie.get("overview", "No synopsis available.")
                        if len(overview) > 160:
                            truncated = overview[:157].rsplit(" ", 1)[0]
                            overview = f"{truncated}..."
                        st.markdown(f'<div class="card-overview-text"><span style="color:#A3A9B8;">📖 </span>{overview}</div>', unsafe_allow_html=True)

    else:
        # "Choose from List" or "Manual Search" Modes
        if not selected_title or not selected_title.strip():
            st.info("👆 Please select or enter a movie title from the sidebar to view recommendations.")
            return

        cleaned_input = selected_title.strip().lower()
        matching_rows = movies_df[movies_df["title"].str.strip().str.lower() == cleaned_input]

        # Selected Movie Hero Card
        if not matching_rows.empty:
            curr_movie = matching_rows.iloc[0]
            rating_html = ""
            if has_ratings and pd.notna(curr_movie.get("vote_average")) and float(curr_movie["vote_average"]) > 0:
                rating_val = float(curr_movie["vote_average"])
                votes_val = int(curr_movie.get("vote_count", 0))
                rating_html = f'<span class="rating-pill">⭐ {rating_val:.1f} / 10</span> <span class="vote-count-text">({votes_val:,} votes)</span>'

            genre_html = format_genres_html(curr_movie.get("genres", ""))
            overview_text = curr_movie.get("overview") if curr_movie.get("overview") else "No synopsis available."

            st.markdown(
                f"""
                <div class="hero-movie-card">
                    <div style="font-size: 0.82rem; font-weight: 600; color: #8B7CFF; margin-bottom: 6px;">📽️ Selected Movie</div>
                    <div class="hero-movie-title">{curr_movie['title']}</div>
                    <div class="badge-row" style="margin-bottom: 8px;">{rating_html}</div>
                    <div class="genre-tag-list" style="margin-bottom: 8px;"><span style="font-size:0.78rem; color:#A3A9B8; margin-right:4px;">🎭</span>{genre_html}</div>
                    <div class="hero-movie-overview"><span style="color:#A3A9B8;">📖 </span>{overview_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(f'<div class="section-heading">📽️ Selected Movie: "{selected_title}"</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="section-heading">✨ Top {top_k} Recommended Movies</div>', unsafe_allow_html=True)

        try:
            recommendations = recommender.recommend(selected_title, top_k=top_k)

            if not recommendations:
                st.warning("No similar titles found.")
                return

            # Display recommendations in structured grid rows
            COLS_PER_ROW = 3
            for row_start in range(0, len(recommendations), COLS_PER_ROW):
                row_recs = recommendations[row_start : row_start + COLS_PER_ROW]
                cols = st.columns(COLS_PER_ROW)
                for idx_in_row, rec in enumerate(row_recs):
                    overall_idx = row_start + idx_in_row
                    with cols[idx_in_row]:
                        with st.container(border=True):
                            st.markdown(
                                f'<div class="movie-title-text">{overall_idx + 1}. {rec["title"]}</div>',
                                unsafe_allow_html=True
                            )

                            similarity_pct = round(rec["similarity_score"] * 100)
                            st.progress(
                                rec["similarity_score"],
                                text=f"Similarity: {rec['similarity_score']:.2f} ({similarity_pct}%)"
                            )

                            genre_html = format_genres_html(rec.get("genres", ""))
                            st.markdown(f'<div class="genre-tag-list" style="margin-top: 8px;"><span style="font-size:0.75rem; color:#A3A9B8; margin-right:4px;">🎭</span>{genre_html}</div>', unsafe_allow_html=True)

                            overview = rec.get("overview", "No synopsis available.")
                            if len(overview) > 160:
                                truncated = overview[:157].rsplit(" ", 1)[0]
                                overview = f"{truncated}..."
                            st.markdown(f'<div class="card-overview-text"><span style="color:#A3A9B8;">📖 </span>{overview}</div>', unsafe_allow_html=True)

        except ValueError as ve:
            st.error(f"⚠️ {str(ve)}")
            st.caption("💡 Try selecting a title from the suggestions dropdown in the sidebar, or check spelling.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
