import pandas as pd
import streamlit as st

from src.evaluation import evaluate_recommender
from src.hybrid_recommendation import collab_df, df, hybrid_recommend, movies


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="movie_camera",
    layout="wide",
)


@st.cache_data
def get_movie_options():
    return movies.sort_values("title")["title"].tolist()


@st.cache_data
def get_user_options():
    return sorted(collab_df.index.astype(int).tolist())


@st.cache_data
def get_genre_options():
    all_genres = set()

    for genres in movies["genres"].dropna():
        all_genres.update(str(genres).split())

    all_genres.discard("no")
    all_genres.discard("genres")
    all_genres.discard("listed")

    return sorted(all_genres)


@st.cache_data
def popular_by_genre(selected_genres, min_ratings, top_n):
    movie_stats = (
        df.groupby(["movieId", "title", "genres"], as_index=False)
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
    )

    if selected_genres:
        selected = set(selected_genres)
        movie_stats = movie_stats[
            movie_stats["genres"].apply(
                lambda genres: selected.issubset(set(str(genres).split()))
            )
        ]

    movie_stats = movie_stats[movie_stats["rating_count"] >= min_ratings].copy()
    movie_stats["score"] = (
        movie_stats["avg_rating"] * movie_stats["rating_count"].pow(0.25)
    )

    return movie_stats.sort_values(
        ["score", "avg_rating", "rating_count"],
        ascending=False,
    ).head(top_n)


@st.cache_data
def get_evaluation_metrics(test_size, eval_top_n, relevance_threshold, hybrid_alpha):
    return evaluate_recommender(
        test_size=test_size,
        top_n=eval_top_n,
        relevance_threshold=relevance_threshold,
        hybrid_alpha=hybrid_alpha,
    )


def render_recommendations(recommendations, selected_genres):
    rows = []
    selected = set(selected_genres)

    for rank, (title, genres, content_score, collab_score, final_score) in enumerate(
        recommendations,
        start=1,
    ):
        if selected and not selected.issubset(set(str(genres).split())):
            continue

        rows.append(
            {
                "Rank": rank,
                "Movie": title,
                "Genres": genres,
                "Content Match": round(content_score, 3),
                "Predicted Rating": round(collab_score, 2),
                "Hybrid Score": round(final_score, 3),
            }
        )

    return pd.DataFrame(rows)


st.title("Movie Recommendation System")
st.caption("Choose your preferences and get recommendations from the hybrid model.")

movie_options = get_movie_options()
user_options = get_user_options()
genre_options = get_genre_options()

with st.sidebar:
    st.header("Your Preferences")

    user_id = st.selectbox(
        "User ID",
        user_options,
        index=user_options.index(4) if 4 in user_options else 0,
    )

    favorite_movie = st.selectbox(
        "Favorite movie",
        movie_options,
        index=movie_options.index("Toy Story (1995)")
        if "Toy Story (1995)" in movie_options
        else 0,
    )

    selected_genres = st.multiselect(
        "Preferred genres",
        genre_options,
        default=[],
    )

    alpha = st.slider(
        "Content-based weight",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="Higher values favor movies similar to the favorite movie. Lower values favor the selected user's predicted ratings.",
    )

    top_n = st.slider("Number of recommendations", 5, 20, 10)
    min_ratings = st.slider("Minimum ratings for genre picks", 1, 100, 20)

tab_hybrid, tab_genres, tab_evaluation, tab_data = st.tabs(
    ["Hybrid Recommendations", "Genre Picks", "Evaluation", "Dataset"]
)

with tab_hybrid:
    st.subheader("Hybrid Recommendations")

    try:
        recommendations = hybrid_recommend(
            user_id=user_id,
            title=favorite_movie,
            alpha=alpha,
            top_n=max(top_n * 3, 20),
        )
        rec_df = render_recommendations(recommendations, selected_genres).head(top_n)

        if rec_df.empty:
            st.info(
                "No hybrid results matched every selected genre. Try fewer genre filters."
            )
        else:
            st.dataframe(
                rec_df,
                hide_index=True,
                use_container_width=True,
            )
    except ValueError as exc:
        st.error(str(exc))

with tab_genres:
    st.subheader("Popular Movies Matching Your Genres")

    genre_df = popular_by_genre(selected_genres, min_ratings, top_n)

    if genre_df.empty:
        st.info("No movies matched these genre and rating-count filters.")
    else:
        st.dataframe(
            genre_df[
                ["title", "genres", "avg_rating", "rating_count", "score"]
            ].rename(
                columns={
                    "title": "Movie",
                    "genres": "Genres",
                    "avg_rating": "Average Rating",
                    "rating_count": "Rating Count",
                    "score": "Popularity Score",
                }
            ).round(
                {
                    "Average Rating": 2,
                    "Popularity Score": 2,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

with tab_evaluation:
    st.subheader("Evaluation Metrics")

    col_test_size, col_top_n, col_threshold, col_alpha = st.columns(4)

    with col_test_size:
        test_size = st.slider(
            "Test split",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
        )

    with col_top_n:
        eval_top_n = st.slider(
            "Top-N recommendations",
            min_value=5,
            max_value=20,
            value=10,
        )

    with col_threshold:
        relevance_threshold = st.slider(
            "Relevant rating threshold",
            min_value=3.0,
            max_value=5.0,
            value=4.0,
            step=0.5,
        )

    with col_alpha:
        eval_alpha = st.slider(
            "Hybrid content weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

    if st.button("Run evaluation", type="primary"):
        with st.spinner("Evaluating recommender performance..."):
            st.session_state["evaluation_results"] = get_evaluation_metrics(
                test_size,
                eval_top_n,
                relevance_threshold,
                eval_alpha,
            )

    if "evaluation_results" not in st.session_state:
        st.info("Run the evaluation to compare all models on the train/test split.")
    else:
        comparison_df, evaluation_settings = st.session_state["evaluation_results"]
        best_model = comparison_df.sort_values("F1-Score", ascending=False).iloc[0]
        summary_rmse, summary_mae, summary_f1 = st.columns(3)

        summary_rmse.metric(
            "Best RMSE",
            f"{comparison_df['RMSE'].min():.3f}",
        )
        summary_mae.metric(
            "Best MAE",
            f"{comparison_df['MAE'].min():.3f}",
        )
        summary_f1.metric(
            "Best F1 Model",
            best_model["Model"],
            f"{best_model['F1-Score']:.3f}",
        )

        st.dataframe(
            comparison_df.round(
                {
                    "RMSE": 3,
                    "MAE": 3,
                    "Precision": 3,
                    "Recall": 3,
                    "F1-Score": 3,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {"Setting": setting, "Value": value}
                    for setting, value in evaluation_settings.items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

with tab_data:
    st.subheader("Dataset Overview")

    col_movies, col_users, col_ratings = st.columns(3)

    col_movies.metric("Movies", f"{movies['movieId'].nunique():,}")
    col_users.metric("Users", f"{df['userId'].nunique():,}")
    col_ratings.metric("Ratings", f"{len(df):,}")

    st.dataframe(
        movies.sort_values("title").head(100),
        hide_index=True,
        use_container_width=True,
    )
