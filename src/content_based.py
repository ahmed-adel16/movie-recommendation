import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Load data
# =========================

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_movies.csv"
df = pd.read_csv(DATA_PATH)


# =========================
# Movies table
# =========================

movies = df[['movieId', 'title', 'combined_features']].drop_duplicates().reset_index(drop=True)


# =========================
# TF-IDF + similarity matrix
# =========================

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

similarity_matrix = cosine_similarity(tfidf_matrix)


# =========================
# Mapping
# =========================

movie_index = pd.Series(movies.index, index=movies["movieId"])


# =========================
# USER-AWARE CONTENT SCORING
# =========================

def get_content_scores(user_id, ratings_df=None):
    """
    Returns: dict(movieId → score)
    """

    if ratings_df is None:
        ratings_df = df

    user_data = ratings_df[ratings_df.userId == user_id]

    if user_data.empty:
        return {}

    # take positively rated movies only
    liked = user_data[user_data.rating >= 4.0]

    if liked.empty:
        liked = user_data

    liked_indices = [
        movie_index[mid]
        for mid in liked.movieId
        if mid in movie_index
    ]

    if not liked_indices:
        return {}

    # aggregate similarity
    sim_scores = similarity_matrix[:, liked_indices].mean(axis=1)

    return {
        movies.iloc[i].movieId: float(score)
        for i, score in enumerate(sim_scores)
    }


# =========================
# Debug
# =========================

if __name__ == "__main__":

    sample_user = df.userId.iloc[0]

    print(f"Sample User ID: {sample_user}")

    scores = get_content_scores(sample_user)

    print("\nTop Content Scores Sample:\n")

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

    for movie_id, score in top:
        title = movies[movies.movieId == movie_id].title.values[0]
        print(movie_id, title, score)