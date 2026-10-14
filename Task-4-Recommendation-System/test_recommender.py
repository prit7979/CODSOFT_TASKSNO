"""
Unit tests for the Movie Recommendation System core engine and dataset loader.
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from recommender import (
    MovieRecommender,
    get_movie_recommendations,
    load_movie_dataset,
    get_fallback_dataset,
    clean_and_prepare_dataset,
    _parse_tmdb_json_field,
    get_movie_suggestions,
    extract_unique_genres,
    explore_by_category,
)


class TestMovieRecommender(unittest.TestCase):
    def setUp(self):
        """Create a representative synthetic dataset for testing."""
        self.sample_data = pd.DataFrame([
            {
                "title": "The Matrix",
                "genres": "Action Sci-Fi",
                "keywords": "virtual reality simulation cyberpunk keanu",
                "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality."
            },
            {
                "title": "The Matrix Reloaded",
                "genres": "Action Sci-Fi",
                "keywords": "virtual reality simulation machine war zion neo",
                "overview": "Freedom fighters struggle against the machine army while Neo seeks answers."
            },
            {
                "title": "Inception",
                "genres": "Action Sci-Fi Thriller",
                "keywords": "dreams subconscious heist reality mind",
                "overview": "A thief who steals corporate secrets through dream-sharing technology is given an inverse task."
            },
            {
                "title": "The Notebook",
                "genres": "Romance Drama",
                "keywords": "love relationship summer diary memory",
                "overview": "A poor yet passionate young man falls in love with a rich young woman."
            },
            {
                "title": "Pride and Prejudice",
                "genres": "Romance Drama",
                "keywords": "love marriage class england regency",
                "overview": "Sparks fly when spirited Elizabeth Bennet meets single, rich, and proud Mr. Darcy."
            },
            {
                "title": "Toy Story",
                "genres": "Animation Adventure Comedy",
                "keywords": "toys friendship childhood woody buzz",
                "overview": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him."
            },
            {
                "title": "Finding Nemo",
                "genres": "Animation Adventure Family",
                "keywords": "ocean fish clownfish lost father son",
                "overview": "After his son is captured in the Great Barrier Reef, a timid clownfish sets out on a journey."
            }
        ])

        self.recommender = MovieRecommender()
        self.recommender.fit(self.sample_data)

    def test_basic_recommendation_similarity(self):
        """Test that sci-fi movie gets another sci-fi movie as top recommendation."""
        recs = self.recommender.recommend("The Matrix", top_k=2)
        self.assertEqual(len(recs), 2)
        titles = [r["title"] for r in recs]
        self.assertEqual(titles[0], "The Matrix Reloaded")
        self.assertNotIn("The Matrix", titles)
        self.assertGreater(recs[0]["similarity_score"], 0.0)
        self.assertLessEqual(recs[0]["similarity_score"], 1.0)

    def test_self_exclusion(self):
        """Verify recommendations strictly exclude the queried movie itself."""
        for title in ["The Matrix", "Inception", "The Notebook", "Toy Story"]:
            recs = self.recommender.recommend(title, top_k=3)
            rec_titles = [r["title"] for r in recs]
            self.assertNotIn(title, rec_titles)

    def test_memory_conscious_no_dense_matrix(self):
        """Verify that fit does not allocate an N x N dense similarity matrix."""
        self.assertIsNone(self.recommender.cosine_sim)
        self.assertTrue(sp.issparse(self.recommender.tfidf_matrix))

    def test_top_k_parameter(self):
        """Test that the number of returned recommendations matches top_k."""
        recs_3 = self.recommender.recommend("Inception", top_k=3)
        self.assertEqual(len(recs_3), 3)

        recs_5 = self.recommender.recommend("Inception", top_k=5)
        self.assertEqual(len(recs_5), 5)

    def test_case_insensitive_and_whitespace_title(self):
        """Test that querying is resilient to lowercase, uppercase, and leading/trailing spaces."""
        recs_normal = self.recommender.recommend("The Matrix", top_k=2)
        recs_lower = self.recommender.recommend("  the matrix  ", top_k=2)
        recs_upper = self.recommender.recommend("THE MATRIX", top_k=2)

        self.assertEqual(
            [r["title"] for r in recs_normal],
            [r["title"] for r in recs_lower]
        )
        self.assertEqual(
            [r["title"] for r in recs_normal],
            [r["title"] for r in recs_upper]
        )

    def test_unknown_title_raises_value_error(self):
        """Test that an unlisted title safely raises a ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.recommender.recommend("Non Existent Movie Title 12345")
        self.assertIn("not found in the dataset", str(ctx.exception))

    def test_empty_or_invalid_title_input(self):
        """Test that empty or non-string titles raise ValueError."""
        with self.assertRaises(ValueError):
            self.recommender.recommend("")
        with self.assertRaises(ValueError):
            self.recommender.recommend(None)  # type: ignore

    def test_missing_and_nan_metadata_handling(self):
        """Test that missing or NaN fields in metadata do not cause failure."""
        faulty_data = pd.DataFrame([
            {
                "title": "Movie A",
                "genres": np.nan,
                "keywords": None,
                "overview": np.nan
            },
            {
                "title": "Movie B",
                "genres": "Action",
                "keywords": np.nan,
                "overview": "Just an action flick."
            },
            {
                "title": "Movie C",
                "genres": "Action",
                "keywords": "explosions",
                "overview": None
            }
        ])

        recommender = MovieRecommender()
        recommender.fit(faulty_data)
        recs = recommender.recommend("Movie A", top_k=2)
        self.assertEqual(len(recs), 2)
        self.assertNotIn("Movie A", [r["title"] for r in recs])

    def test_duplicate_titles_handling(self):
        """Test that duplicate titles are safely deduplicated."""
        dup_data = pd.DataFrame([
            {"title": "Avatar", "genres": "Sci-Fi", "overview": "Pandora exploration"},
            {"title": "Avatar", "genres": "Sci-Fi Action", "overview": "Another release entry"},
            {"title": "Aliens", "genres": "Sci-Fi Horror", "overview": "Space xenomorphs"},
        ])
        recommender = MovieRecommender()
        recommender.fit(dup_data)
        self.assertEqual(len(recommender.df), 2)
        recs = recommender.recommend("Avatar", top_k=1)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["title"], "Aliens")

    def test_unfitted_recommender_raises_runtime_error(self):
        """Test that calling recommend before fit raises a RuntimeError."""
        unfitted = MovieRecommender()
        with self.assertRaises(RuntimeError):
            unfitted.recommend("The Matrix")

    def test_convenience_function(self):
        """Test the standalone get_movie_recommendations function."""
        recs = get_movie_recommendations(self.sample_data, "The Notebook", top_k=2)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["title"], "Pride and Prejudice")


class TestDatasetLoading(unittest.TestCase):
    def test_load_real_movie_dataset(self):
        """Test that the local data/movies.csv loads with required columns."""
        df = load_movie_dataset()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        for col in ["title", "genres", "keywords", "overview"]:
            self.assertIn(col, df.columns)
        self.assertGreaterEqual(len(df), 1000)

    def test_load_dataset_missing_file_raises_error(self):
        """Test that a non-existent file path raises FileNotFoundError by default."""
        with self.assertRaises(FileNotFoundError):
            load_movie_dataset("non_existent_folder/fake_movies.csv", fallback_on_missing=False)

    def test_load_dataset_missing_file_fallback(self):
        """Test that fallback_on_missing=True returns fallback dataframe."""
        df_fb = load_movie_dataset("non_existent_folder/fake_movies.csv", fallback_on_missing=True)
        self.assertIsInstance(df_fb, pd.DataFrame)
        self.assertIn("The Matrix", df_fb["title"].values)

    def test_parse_tmdb_json_field(self):
        """Test parsing of TMDB JSON string lists."""
        json_str = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
        parsed = _parse_tmdb_json_field(json_str)
        self.assertEqual(parsed, "Action, Adventure")

        # Test plain strings and NaNs
        self.assertEqual(_parse_tmdb_json_field("Action, Comedy"), "Action, Comedy")
        self.assertEqual(_parse_tmdb_json_field(None), "")
        self.assertEqual(_parse_tmdb_json_field(np.nan), "")

    def test_clean_and_prepare_dataset_synthetic(self):
        """Test merging metadata and keywords with invalid IDs, duplicate titles, and JSON formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mm_file = os.path.join(tmpdir, "test_mm.csv")
            kw_file = os.path.join(tmpdir, "test_kw.csv")
            out_file = os.path.join(tmpdir, "test_out.csv")

            # Synthetic movies_metadata with 1 invalid ID, 1 duplicate title, 1 missing title
            mm_data = pd.DataFrame([
                {"id": "101", "title": "Inception", "genres": '[{"name": "Sci-Fi"}]', "overview": "Dream heist", "vote_count": "5000"},
                {"id": "102", "title": "Interstellar", "genres": '[{"name": "Adventure"}]', "overview": "Space travel", "vote_count": "4000"},
                {"id": "103", "title": "Inception", "genres": '[{"name": "Action"}]', "overview": "Duplicate entry", "vote_count": "100"},
                {"id": "invalid_date_id", "title": "Bad Movie", "genres": "[]", "overview": "Corrupt ID", "vote_count": "10"},
                {"id": "104", "title": None, "genres": "[]", "overview": "No title", "vote_count": "0"},
            ])
            mm_data.to_csv(mm_file, index=False)

            # Synthetic keywords
            kw_data = pd.DataFrame([
                {"id": "101", "keywords": '[{"name": "dreams"}, {"name": "subconscious"}]'},
                {"id": "102", "keywords": '[{"name": "wormhole"}, {"name": "black hole"}]'},
                {"id": "105", "keywords": '[{"name": "orphan"}]'},
            ])
            kw_data.to_csv(kw_file, index=False)

            cleaned_df = clean_and_prepare_dataset(mm_file, kw_file, out_file)

            # Should retain exactly 2 valid movies: Inception and Interstellar
            self.assertEqual(len(cleaned_df), 2)
            self.assertIn("Inception", cleaned_df["title"].values)
            self.assertIn("Interstellar", cleaned_df["title"].values)
            # Check keywords merged properly
            inc_row = cleaned_df[cleaned_df["title"] == "Inception"].iloc[0]
            self.assertIn("dreams", inc_row["keywords"])
            # Check genres parsed
            self.assertEqual(inc_row["genres"], "Sci-Fi")
            # Verify file was written
            self.assertTrue(os.path.exists(out_file))


class TestMovieSuggestions(unittest.TestCase):
    def setUp(self):
        self.catalog_titles = [
            "Inception",
            "The Dark Knight",
            "Spider-Man",
            "Spider-Man 2",
            "Spider-Man 3",
            "The Amazing Spider-Man",
            "Batman Begins",
            "Batman",
            "Batman Returns",
            "Batman Forever",
            "Battleship",
            "Spirited Away",
            "Toy Story",
        ]

    def test_suggestions_for_spi(self):
        """Test suggestions for 'spi' include Spider-Man titles."""
        matches = get_movie_suggestions("spi", self.catalog_titles, max_suggestions=8)
        self.assertGreater(len(matches), 0)
        self.assertIn("Spider-Man", matches)
        self.assertIn("Spider-Man 2", matches)
        self.assertLessEqual(len(matches), 8)

    def test_suggestions_for_bat(self):
        """Test suggestions for 'bat' include Batman titles."""
        matches = get_movie_suggestions("bat", self.catalog_titles, max_suggestions=8)
        self.assertGreater(len(matches), 0)
        self.assertIn("Batman", matches)
        self.assertIn("Batman Begins", matches)
        self.assertLessEqual(len(matches), 8)

    def test_suggestions_no_match(self):
        """Test query with no matching titles returns empty list."""
        matches = get_movie_suggestions("nonexistent123xyz", self.catalog_titles)
        self.assertEqual(matches, [])

    def test_suggestions_case_and_whitespace(self):
        """Test suggestions ignore leading/trailing whitespace and case."""
        matches_upper = get_movie_suggestions("  SPI  ", self.catalog_titles)
        matches_lower = get_movie_suggestions("spi", self.catalog_titles)
        self.assertEqual(matches_upper, matches_lower)

    def test_suggestions_max_limit(self):
        """Test that returned suggestions never exceed max_suggestions."""
        matches = get_movie_suggestions("a", self.catalog_titles, max_suggestions=4)
        self.assertEqual(len(matches), 4)


class TestExploreByCategory(unittest.TestCase):
    def setUp(self):
        self.sample_category_df = pd.DataFrame([
            {
                "title": "Action Hero",
                "genres": "Action",
                "vote_average": 7.5,
                "vote_count": 500,
                "overview": "A solo action film."
            },
            {
                "title": "Space Battles",
                "genres": "Action, Science Fiction",
                "vote_average": 8.5,
                "vote_count": 1500,
                "overview": "Space warfare with high stakes."
            },
            {
                "title": "Future Cyberpunk",
                "genres": "Action, Science Fiction, Thriller",
                "vote_average": 8.0,
                "vote_count": 2000,
                "overview": "Cybernetic action thriller."
            },
            {
                "title": "Love in Paris",
                "genres": "Romance, Drama",
                "vote_average": 7.8,
                "vote_count": 300,
                "overview": "Romantic strolls in Paris."
            },
            {
                "title": "Indie Low Rated",
                "genres": "Action",
                "vote_average": 4.2,
                "vote_count": 100,
                "overview": "Low rated action film."
            },
            {
                "title": "Obscure Unvoted",
                "genres": "Action, Science Fiction",
                "vote_average": 9.5,
                "vote_count": 2,  # Very few votes
                "overview": "Unvoted indie."
            }
        ])

    def test_extract_unique_genres(self):
        """Test that unique genres are extracted and sorted."""
        genres = extract_unique_genres(self.sample_category_df)
        self.assertEqual(
            genres,
            ["Action", "Drama", "Romance", "Science Fiction", "Thriller"]
        )

    def test_single_genre_selection(self):
        """Test filtering by a single genre returns only matching films."""
        recs = explore_by_category(self.sample_category_df, selected_genres=["Romance"], top_k=5)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["title"], "Love in Paris")
        self.assertIn("Romance", recs[0]["genres"])

    def test_multiple_genres_ranking(self):
        """Test films matching multiple selected genres rank ahead of single-match films."""
        recs = explore_by_category(
            self.sample_category_df,
            selected_genres=["Action", "Science Fiction"],
            top_k=5
        )
        self.assertGreater(len(recs), 0)
        # First recommendation should match both genres (match_count == 2)
        self.assertEqual(recs[0]["match_count"], 2)
        self.assertIn(recs[0]["title"], ["Space Battles", "Future Cyberpunk", "Obscure Unvoted"])

    def test_minimum_rating_filtering(self):
        """Test that min_rating excludes films below the threshold and filters low votes."""
        recs = explore_by_category(
            self.sample_category_df,
            selected_genres=["Action"],
            min_rating=7.0,
            min_vote_count=20,
            top_k=10
        )
        # Should exclude 'Indie Low Rated' (4.2 < 7.0)
        titles = [r["title"] for r in recs]
        self.assertNotIn("Indie Low Rated", titles)
        # All returned titles should have rating >= 7.0
        for r in recs:
            self.assertGreaterEqual(r["rating"], 7.0)

    def test_recommendation_count_limit(self):
        """Test that returned results never exceed top_k."""
        recs = explore_by_category(
            self.sample_category_df,
            selected_genres=["Action"],
            top_k=2
        )
        self.assertEqual(len(recs), 2)

    def test_empty_results_no_matching_genre(self):
        """Test that selecting a non-existent genre returns an empty list."""
        recs = explore_by_category(
            self.sample_category_df,
            selected_genres=["NonExistentGenre"],
            top_k=5
        )
        self.assertEqual(recs, [])

    def test_empty_results_empty_genres_input(self):
        """Test that passing an empty list of genres returns an empty list."""
        recs = explore_by_category(self.sample_category_df, selected_genres=[], top_k=5)
        self.assertEqual(recs, [])


if __name__ == "__main__":
    unittest.main()
