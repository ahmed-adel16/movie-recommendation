import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Load processed dataset
# =========================

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_movies.csv"

df = pd.read_csv(DATA_PATH)


# =========================
# Remove duplicate movies
# =========================

movies = df[['movieId', 'title', 'genres']].drop_duplicates().reset_index(drop=True)


# =========================
# TF-IDF Vectorization
# =========================

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(
    movies['genres']
)


# =========================
# Cosine Similarity
# =========================

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)


# =========================
# Create title index mapping
# =========================

indices = pd.Series(
    movies.index,
    index=movies['title']
).drop_duplicates()


# =========================
# Recommendation Function
# =========================

def content_recommendations(title, top_n=10):
    if title not in indices.index:
        raise ValueError(f"Unknown movie title: {title}")

    # get movie index
    idx = indices[title]

    # similarity scores
    sim_scores = list(
        enumerate(cosine_sim[idx])
    )

    # sort descending
    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # remove same movie
    sim_scores = sim_scores[1:top_n + 1]

    # get movie indices
    movie_indices = [
        i[0]
        for i in sim_scores
    ]

    # return titles
    return movies['title'].iloc[
        movie_indices
    ]


if __name__ == "__main__":
    recommendations = content_recommendations(
        "Toy Story (1995)"
    )

    print("\nRecommended Movies:\n")

    for movie in recommendations:
        print(movie)
