import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


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

# optional noise filtering
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

movies = movies.merge(movie_tags, on="movieId", how="left")
movies["tag"] = movies["tag"].fillna("")


# =========================
# Content Features (IMPROVED)
# =========================

movies["combined_features"] = (
    movies["genres"].fillna("").str.replace("|", " ") +
    " " +
    movies["tag"]
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

content_sim = cosine_similarity(tfidf_matrix)


# =========================
# Mapping
# =========================

movieid_to_idx = pd.Series(
    movies.index,
    index=movies["movieId"]
)


# =========================
# Collaborative Filtering (SVD)
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

svd = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

X_reduced = svd.fit_transform(X)
X_reconstructed = np.dot(X_reduced, svd.components_)

collab_pred = X_reconstructed + user_mean.values.reshape(-1, 1)

collab_df = pd.DataFrame(
    collab_pred,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
).fillna(0)


# =========================
# HYBRID FUNCTION (FINAL)
# =========================

def hybrid_recommend(user_id, movie_id=None, alpha=0.5, top_n=10):

    if user_id not in collab_df.index:
        raise ValueError(f"Unknown user ID: {user_id}")

    # =========================
    # 1. USER BASED SCORES (GLOBAL)
    # =========================

    user_scores = collab_df.loc[user_id]

    # normalize to [0,1]
    user_norm = (user_scores / 5.0).clip(0, 1)

    results = []

    # =========================
    # 2. CASE A: NO SEED MOVIE (PURE USER MODE)
    # =========================

    if movie_id is None:

        for movie in movies.itertuples():

            mid = movie.movieId

            if mid not in user_norm.index:
                continue

            final_score = float(user_norm[mid])

            results.append((
                mid,
                movie.title,
                movie.genres,
                0.0,                 # no content seed
                float(user_scores[mid]),
                final_score
            ))

    # =========================
    # 3. CASE B: WITH SEED MOVIE (SOFT BOOST MODE)
    # =========================

    else:

        if movie_id not in movieid_to_idx.index:
            raise ValueError(f"Unknown movie ID: {movie_id}")

        idx = movieid_to_idx[movie_id]

        sim_scores = list(enumerate(content_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n * 3]

        for i, content_score in sim_scores:

            target_movie_id = movies.iloc[i]['movieId']

            if target_movie_id not in user_norm.index:
                continue

            collab_score = user_norm[target_movie_id]

            final_score = (
                alpha * content_score +
                (1 - alpha) * collab_score
            )

            results.append((
                target_movie_id,
                movies.iloc[i]['title'],
                movies.iloc[i]['genres'],
                float(content_score),
                float(user_scores[target_movie_id]),
                float(final_score)
            ))

    # =========================
    # 4. SORT + RETURN
    # =========================

    results.sort(key=lambda x: x[5], reverse=True)

    return results[:top_n]


# =========================
# TEST
# =========================

if __name__ == "__main__":

    user_id = df.userId.iloc[0]
    movie_id = movies.movieId.iloc[0]

    print(f"User: {user_id}")
    print(f"Seed Movie: {movies.iloc[0]['title']}")

    recs = hybrid_recommend(user_id, movie_id, alpha=0.6)

    print("\nTop Hybrid Recommendations:\n")

    for r in recs:
        print(r)