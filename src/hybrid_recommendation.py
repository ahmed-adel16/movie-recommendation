import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# =========================
# Load data
# =========================

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_movies.csv"

df = pd.read_csv(DATA_PATH)


# =========================
# Movies table (unique)
# =========================

movies = df[['movieId', 'title', 'genres']].drop_duplicates().reset_index(drop=True)


# =========================
# CONTENT-BASED PART
# =========================

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movies['genres'])

content_sim = cosine_similarity(tfidf_matrix)


indices = pd.Series(
    movies.index,
    index=movies['title']
).drop_duplicates()


# =========================
# COLLABORATIVE PART (SVD)
# =========================

user_movie_matrix = df.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

user_mean = user_movie_matrix.mean(axis=1)

matrix_centered = user_movie_matrix.sub(user_mean, axis=0).fillna(0)

X = matrix_centered.values

n_components = min(50, max(1, min(X.shape) - 1))
svd = TruncatedSVD(n_components=n_components, random_state=42)

X_reduced = svd.fit_transform(X)

X_reconstructed = np.dot(X_reduced, svd.components_)

collab_pred = X_reconstructed + user_mean.values.reshape(-1, 1)

collab_df = pd.DataFrame(
    collab_pred,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)


# =========================
# HYBRID FUNCTION
# =========================

def hybrid_recommend(user_id, title, alpha=0.5, top_n=10):
    if user_id not in collab_df.index:
        raise ValueError(f"Unknown user ID: {user_id}")

    if title not in indices.index:
        raise ValueError(f"Unknown movie title: {title}")

    idx = indices[title]

    # content similarity
    sim_scores = list(enumerate(content_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n + 1]

    results = []

    for i, content_score in sim_scores:

        movie_id = movies.iloc[i]['movieId']

        # collaborative score
        if movie_id not in collab_df.columns:
            continue

        collab_score = collab_df.loc[user_id, movie_id]

        # normalization (important)
        collab_norm = collab_score / 5.0

        final_score = (
            alpha * content_score +
            (1 - alpha) * collab_norm
        )

        results.append((
            movies.iloc[i]['title'],
            movies.iloc[i]['genres'],
            float(content_score),
            float(collab_score),
            float(final_score),
        ))

    results.sort(key=lambda x: x[4], reverse=True)

    return results


if __name__ == "__main__":
    user_id = 4
    movie_title = "Adventures of Robin Hood, The (1938)"

    recs = hybrid_recommend(user_id, movie_title)

    print("\nHybrid Recommendations:\n")

    for movie, genres, content_score, collab_score, final_score in recs:
        print(movie, genres, content_score, collab_score, final_score)
