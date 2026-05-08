from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_movies.csv"


def load_ratings():
    columns = ["userId", "movieId", "rating", "title", "genres"]
    return pd.read_csv(DATA_PATH, usecols=columns)


def split_ratings(ratings, test_size=0.2):
    return train_test_split(
        ratings,
        test_size=test_size,
        random_state=42,
        stratify=ratings["userId"],
    )


def build_movie_table(ratings):
    return ratings[["movieId", "title", "genres"]].drop_duplicates().reset_index(drop=True)


def build_collaborative_predictions(train_ratings, n_components=50):
    user_movie_matrix = train_ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
    )

    user_mean = user_movie_matrix.mean(axis=1)
    matrix_centered = user_movie_matrix.sub(user_mean, axis=0).fillna(0)

    x = matrix_centered.values
    n_components = min(n_components, max(1, min(x.shape) - 1))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    x_reduced = svd.fit_transform(x)
    x_reconstructed = np.dot(x_reduced, svd.components_)

    predictions = x_reconstructed + user_mean.values.reshape(-1, 1)

    return pd.DataFrame(
        predictions,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.columns,
    )


def build_content_predictions(train_ratings, movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    similarity = cosine_similarity(tfidf_matrix)

    movie_positions = pd.Series(movies.index, index=movies["movieId"])
    content_predictions = {}

    for user_id, user_rows in train_ratings.groupby("userId"):
        rated_positions = [
            movie_positions[movie_id]
            for movie_id in user_rows["movieId"]
            if movie_id in movie_positions.index
        ]

        if not rated_positions:
            continue

        user_ratings = user_rows[
            user_rows["movieId"].isin(movie_positions.index)
        ]["rating"].to_numpy()

        user_similarity = similarity[:, rated_positions]
        numerator = np.dot(user_similarity, user_ratings)
        denominator = user_similarity.sum(axis=1)

        predicted = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, user_ratings.mean(), dtype=float),
            where=denominator != 0,
        )

        content_predictions[user_id] = pd.Series(
            predicted,
            index=movies["movieId"],
        )

    return pd.DataFrame(content_predictions).T


def build_hybrid_predictions(content_predictions, collaborative_predictions, alpha=0.5):
    shared_users = content_predictions.index.intersection(collaborative_predictions.index)
    shared_movies = content_predictions.columns.intersection(collaborative_predictions.columns)

    content_shared = content_predictions.loc[shared_users, shared_movies]
    collaborative_shared = collaborative_predictions.loc[shared_users, shared_movies]

    return (alpha * content_shared) + ((1 - alpha) * collaborative_shared)


def predict_test_ratings(test_ratings, prediction_matrix, global_mean):
    predicted = []

    for row in test_ratings.itertuples(index=False):
        if row.userId in prediction_matrix.index and row.movieId in prediction_matrix.columns:
            predicted.append(prediction_matrix.loc[row.userId, row.movieId])
        else:
            predicted.append(global_mean)

    return np.array(predicted)


def evaluate_rating_accuracy(test_ratings, predicted_ratings):
    actual = test_ratings["rating"].to_numpy()

    return {
        "RMSE": sqrt(mean_squared_error(actual, predicted_ratings)),
        "MAE": mean_absolute_error(actual, predicted_ratings),
    }


def evaluate_top_n(
    train_ratings,
    test_ratings,
    prediction_matrix,
    top_n=10,
    relevance_threshold=4.0,
):
    relevant_test = test_ratings[test_ratings["rating"] >= relevance_threshold]
    users_with_relevant_items = relevant_test.groupby("userId")["movieId"].apply(set)
    train_items_by_user = train_ratings.groupby("userId")["movieId"].apply(set)

    true_positive = 0
    recommended_total = 0
    relevant_total = 0
    evaluated_users = 0

    for user_id, relevant_items in users_with_relevant_items.items():
        if user_id not in prediction_matrix.index:
            continue

        user_scores = prediction_matrix.loc[user_id].dropna()
        already_seen = train_items_by_user.get(user_id, set())
        user_scores = user_scores.drop(labels=list(already_seen), errors="ignore")

        recommended_items = set(user_scores.sort_values(ascending=False).head(top_n).index)

        if not recommended_items:
            continue

        true_positive += len(recommended_items & relevant_items)
        recommended_total += len(recommended_items)
        relevant_total += len(relevant_items)
        evaluated_users += 1

    precision = true_positive / recommended_total if recommended_total else 0.0
    recall = true_positive / relevant_total if relevant_total else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Evaluated Users": evaluated_users,
    }


def evaluate_model(
    model_name,
    prediction_matrix,
    train_ratings,
    test_ratings,
    global_mean,
    top_n,
    relevance_threshold,
):
    predicted_ratings = predict_test_ratings(
        test_ratings,
        prediction_matrix,
        global_mean=global_mean,
    )

    metrics = evaluate_rating_accuracy(test_ratings, predicted_ratings)
    metrics.update(
        evaluate_top_n(
            train_ratings,
            test_ratings,
            prediction_matrix,
            top_n=top_n,
            relevance_threshold=relevance_threshold,
        )
    )
    metrics["Model"] = model_name

    return metrics


def evaluate_recommender(
    test_size=0.2,
    top_n=10,
    relevance_threshold=4.0,
    n_components=50,
    hybrid_alpha=0.5,
):
    ratings = load_ratings()
    train_ratings, test_ratings = split_ratings(ratings, test_size=test_size)
    movies = build_movie_table(ratings)
    global_mean = train_ratings["rating"].mean()

    content_predictions = build_content_predictions(train_ratings, movies)
    collaborative_predictions = build_collaborative_predictions(
        train_ratings,
        n_components=n_components,
    )
    hybrid_predictions = build_hybrid_predictions(
        content_predictions,
        collaborative_predictions,
        alpha=hybrid_alpha,
    )

    model_outputs = [
        ("Content-Based", content_predictions),
        ("Collaborative SVD", collaborative_predictions),
        ("Hybrid", hybrid_predictions),
    ]

    results = [
        evaluate_model(
            model_name,
            prediction_matrix,
            train_ratings,
            test_ratings,
            global_mean,
            top_n,
            relevance_threshold,
        )
        for model_name, prediction_matrix in model_outputs
    ]

    return pd.DataFrame(results)[
        [
            "Model",
            "RMSE",
            "MAE",
            "Precision",
            "Recall",
            "F1-Score",
            "Evaluated Users",
        ]
    ], {
        "Train Ratings": len(train_ratings),
        "Test Ratings": len(test_ratings),
        "Test Split": test_size,
        "Top N": top_n,
        "Relevant Rating Threshold": relevance_threshold,
        "Hybrid Alpha": hybrid_alpha,
    }


if __name__ == "__main__":
    comparison, settings = evaluate_recommender()

    print("Evaluation Settings")
    print("-------------------")
    for metric, value in settings.items():
        print(f"{metric}: {value}")

    print("\nModel Comparison")
    print("----------------")
    print(comparison.round(4).to_string(index=False))
