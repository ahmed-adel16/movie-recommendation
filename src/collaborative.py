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

movies = df[['movieId', 'title', 'combined_features']].drop_duplicates()
movies = movies.reset_index(drop=True) # reset index to start from 0 after dropping duplicates


# =========================
# TF-IDF Vectorization
# =========================

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(
    movies['combined_features']
    )




# =========================
# Create title index mapping
# =========================

indices = pd.Series(
    movies.index, 
    index=movies['movieId']
    )


# =========================
# Recommendation Function
# =========================

def content_recommendations(movie_id, top_n=10):

    if movie_id not in indices.index:
        raise ValueError("Unknown movieId")

    idx = indices[movie_id]

    # compute similarity only for this movie
    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    )[0]

    # pair index with similarity
    sim_scores = list(enumerate(sim_scores))

    # sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # remove itself
    sim_scores = sim_scores[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices][['movieId', 'title']]


if __name__ == "__main__":

    sample_movie = movies.iloc[0]['movieId']
    print(f"Sample movie: {movies.iloc[0]['title']}")

    recommendations = content_recommendations(sample_movie, 5) # recommendation of top 5 movies 

    print("\nRecommended Movies:\n")

    print(recommendations)