import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Load data
# =========================

BASE_PATH = Path(__file__).resolve().parents[1] / "data"

movies_path = BASE_PATH / "processed_movies.csv"
tags_path = BASE_PATH / "tags.csv"

df = pd.read_csv(movies_path)
tags_df = pd.read_csv(tags_path)


# =========================
# Clean tags
# =========================

tags_df = tags_df.dropna(subset=["tag"])
tags_df["tag"] = tags_df["tag"].astype(str).str.lower().str.replace(" ", "_")

# optional noise reduction
tag_counts = tags_df["tag"].value_counts()
valid_tags = tag_counts[tag_counts >= 5].index
tags_df = tags_df[tags_df["tag"].isin(valid_tags)]


# =========================
# Aggregate tags per movie
# =========================

movie_tags = (
    tags_df.groupby("movieId")["tag"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)


# =========================
# Movies table
# =========================

movies = df[['movieId', 'title', 'genres']].drop_duplicates().reset_index(drop=True)

# merge tags
movies = movies.merge(movie_tags, on="movieId", how="left")
movies["tag"] = movies["tag"].fillna("")


# =========================
# Combined features
# =========================

movies["combined_features"] = (
    movies["genres"].fillna("").str.replace("|", " ") +
    " " +
    movies["tag"]
)


# =========================
# TF-IDF
# =========================

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movies["combined_features"])


# =========================
# Index mapping
# =========================

indices = pd.Series(
    movies.index,
    index=movies["movieId"]
)


# =========================
# Recommendation Function
# =========================

def content_recommendations(movie_id, top_n=10):

    if movie_id not in indices.index:
        raise ValueError("Unknown movieId")

    idx = indices[movie_id]

    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    )[0]

    sim_scores = list(enumerate(sim_scores))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices][['movieId', 'title', 'genres']]


# =========================
# Debug
# =========================

if __name__ == "__main__":

    sample_movie = movies.iloc[0]['movieId']

    print(f"Sample movie: {movies.iloc[0]['title']}")

    recommendations = content_recommendations(sample_movie, 5)

    print("\nRecommended Movies:\n")
    print(recommendations)