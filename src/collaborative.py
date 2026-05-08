import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# Load data
# =========================

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_movies.csv"

df = pd.read_csv(DATA_PATH)


# =========================
# User-Item Matrix
# =========================

user_movie_matrix = df.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)


# =========================
# Mean center (IMPORTANT FIX)
# =========================

user_mean = user_movie_matrix.mean(axis=1)

matrix_centered = user_movie_matrix.sub(user_mean, axis=0)
matrix_centered = matrix_centered.fillna(0)


# =========================
# Convert to numpy
# =========================

X = matrix_centered.values


# =========================
# SIMPLE SVD (NO TUNING)
# =========================

n_components = min(50, max(1, min(X.shape) - 1))
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X)


# =========================
# Reconstruct
# =========================

X_reconstructed = np.dot(X_reduced, svd.components_)


# =========================
# Add back user mean
# =========================

predicted = X_reconstructed + user_mean.values.reshape(-1, 1)


# =========================
# DataFrame
# =========================

pred_df = pd.DataFrame(
    predicted,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)


def evaluate_model():
    mask = ~np.isnan(user_movie_matrix.values)

    actual = user_movie_matrix.values[mask]
    pred = predicted[mask]

    rmse = sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)

    return rmse, mae


def predict_rating(user_id, movie_id):
    if user_id not in pred_df.index:
        raise ValueError(f"Unknown user ID: {user_id}")

    if movie_id not in pred_df.columns:
        raise ValueError(f"Unknown movie ID: {movie_id}")

    return pred_df.loc[user_id, movie_id]


if __name__ == "__main__":
    rmse, mae = evaluate_model()

    print("\nRMSE:", rmse)
    print("MAE:", mae)

    user_id = 4
    movie_id = 1391

    print("\nPrediction Example")
    print("------------------")
    print("User:", user_id)
    print("Movie:", movie_id)

    print("Predicted Rating:", predict_rating(user_id, movie_id))
    print("Actual Rating:", df[
        (df.userId == user_id) &
        (df.movieId == movie_id)
    ]['rating'].values[0])
