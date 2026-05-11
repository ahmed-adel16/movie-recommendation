import pandas as pd
import streamlit as st

from src.evaluation import evaluate_recommender
from src.hybrid_recommendation import (
    collab_df,
    df,
    hybrid_recommend,
    movies,
)

from src.tmdb_api import fetch_movie_details


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown(
    """
    <style>

    .movie-card {
        background-color: #111111;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid #333333;
    }

    .movie-title {
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
        color: white;
    }

    .movie-text {
        font-size: 14px;
        color: #cccccc;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# CACHE FUNCTIONS
# -------------------------


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
        df.groupby(
            ["movieId", "title", "genres"],
            as_index=False
        )
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
    )

    if selected_genres:
        selected = set(selected_genres)

        movie_stats = movie_stats[
            movie_stats["genres"].apply(
                lambda genres:
                selected.issubset(
                    set(str(genres).split())
                )
            )
        ]

    movie_stats = movie_stats[
        movie_stats["rating_count"] >= min_ratings
    ].copy()

    movie_stats["score"] = (
        movie_stats["avg_rating"]
        * movie_stats["rating_count"].pow(0.25)
    )

    return movie_stats.sort_values(
        ["score", "avg_rating", "rating_count"],
        ascending=False,
    ).head(top_n)


@st.cache_data
def get_evaluation_metrics(
    test_size,
    eval_top_n,
    relevance_threshold,
    hybrid_alpha,
):

    return evaluate_recommender(
        test_size=test_size,
        top_n=eval_top_n,
        relevance_threshold=relevance_threshold,
        hybrid_alpha=hybrid_alpha,
    )


# -------------------------
# MOVIE CARD RENDER
# -------------------------


def render_movie_cards(recommendations, selected_genres):

    selected = set(selected_genres)

    filtered = []

    for rank, (
        movie_id,
        title,
        genres,
        content_score,
        collab_score,
        final_score,
    ) in enumerate(recommendations, start=1):

        if selected and not selected.issubset(set(str(genres).split())):
            continue

        filtered.append(
            {
                "movie_id": movie_id,   # 🔥 IMPORTANT FIX
                "title": title,
                "genres": genres,
                "content_score": content_score,
                "collab_score": collab_score,
                "final_score": final_score,
            }
        )

    if not filtered:
        st.info("No hybrid results matched filters.")
        return

    cols = st.columns(4)

    for idx, movie in enumerate(filtered):

        # 🔥 FIX: USE MOVIE ID NOT TITLE
        tmdb = fetch_movie_details(movie["movie_id"])

        poster = None
        overview = ""

        if tmdb:
            poster = tmdb.get("poster")
            overview = tmdb.get("overview", "")

        with cols[idx % 4]:

            with st.container(border=True):

                # 🔥 FIX: force fallback image if None or empty
                if poster and isinstance(poster, str) and len(poster) > 10:
                    st.image(poster, use_container_width=True)
                else:
                    st.image(
                        "https://via.placeholder.com/500x750?text=No+Poster",
                        use_container_width=True,
                    )

                st.markdown(
                    f"<div class='movie-title'>{movie['title']}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='movie-text'>🎭 {movie['genres']}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='movie-text'>⭐ Hybrid: {movie['final_score']:.3f}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='movie-text'>🧠 Content: {movie['content_score']:.3f}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='movie-text'>👥 Collaborative: {movie['collab_score']:.2f}</div>",
                    unsafe_allow_html=True,
                )

                if overview:
                    st.caption(overview[:120] + "...")

# -------------------------
# TITLE
# -------------------------

st.title("🎬 Movie Recommendation System")

st.caption(
    "Hybrid recommendation system using collaborative + content-based filtering"
)

# -------------------------
# SIDEBAR
# -------------------------

movie_options = get_movie_options()
user_options = get_user_options()
genre_options = get_genre_options()

with st.sidebar:

    st.header("Preferences")

    user_id = st.selectbox(
        "User ID",
        user_options,
        index=user_options.index(4)
        if 4 in user_options
        else 0,
    )

    favorite_movie = st.selectbox(
        "Favorite Movie",
        movie_options,
        index=movie_options.index(
            "Toy Story (1995)"
        )
        if "Toy Story (1995)" in movie_options
        else 0,
    )

    selected_genres = st.multiselect(
        "Preferred Genres",
        genre_options,
        default=[],
    )

    alpha = st.slider(
        "Content-Based Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
    )

    top_n = st.slider(
        "Number of Recommendations",
        5,
        20,
        10,
    )

    min_ratings = st.slider(
        "Minimum Ratings",
        1,
        100,
        20,
    )

# -------------------------
# TABS
# -------------------------

tab_hybrid, tab_genres, tab_evaluation, tab_data = st.tabs(
    [
        "Hybrid Recommendations",
        "Genre Picks",
        "Evaluation",
        "Dataset",
    ]
)

# -------------------------
# HYBRID TAB
# -------------------------

with tab_hybrid:

    st.subheader("Recommended Movies")

    try:

        movie_id = movies[movies["title"] == favorite_movie]["movieId"].values[0]

        recommendations = hybrid_recommend(
            user_id=user_id,
            movie_id=movie_id,
            alpha=alpha,
            top_n=max(top_n * 3, 20),
        )

        render_movie_cards(
            recommendations[:top_n],
            selected_genres,
        )

    except ValueError as exc:
        st.error(str(exc))

# -------------------------
# GENRE TAB
# -------------------------

with tab_genres:

    st.subheader(
        "Popular Movies Matching Genres"
    )

    genre_df = popular_by_genre(
        selected_genres,
        min_ratings,
        top_n,
    )

    if genre_df.empty:

        st.info(
            "No movies matched these filters."
        )

    else:

        st.dataframe(
            genre_df[
                [
                    "title",
                    "genres",
                    "avg_rating",
                    "rating_count",
                    "score",
                ]
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

# -------------------------
# EVALUATION TAB
# -------------------------

with tab_evaluation:

    st.subheader("Evaluation Metrics")

    col_test_size, col_top_n, col_threshold, col_alpha = st.columns(4)

    with col_test_size:

        test_size = st.slider(
            "Test split",
            0.1,
            0.4,
            0.2,
            0.05,
        )

    with col_top_n:

        eval_top_n = st.slider(
            "Top-N",
            5,
            20,
            10,
        )

    with col_threshold:

        relevance_threshold = st.slider(
            "Threshold",
            3.0,
            5.0,
            4.0,
            0.5,
        )

    with col_alpha:

        eval_alpha = st.slider(
            "Hybrid Weight",
            0.0,
            1.0,
            0.5,
            0.05,
        )

    if st.button(
        "Run Evaluation",
        type="primary",
    ):

        with st.spinner(
            "Evaluating..."
        ):

            st.session_state[
                "evaluation_results"
            ] = get_evaluation_metrics(
                test_size,
                eval_top_n,
                relevance_threshold,
                eval_alpha,
            )

    if "evaluation_results" in st.session_state:

        comparison_df, evaluation_settings = (
            st.session_state["evaluation_results"]
        )

        st.dataframe(
            comparison_df.round(3),
            hide_index=True,
            use_container_width=True,
        )

# -------------------------
# DATASET TAB
# -------------------------

with tab_data:

    st.subheader("Dataset Overview")

    col_movies, col_users, col_ratings = st.columns(3)

    col_movies.metric(
        "Movies",
        f"{movies['movieId'].nunique():,}",
    )

    col_users.metric(
        "Users",
        f"{df['userId'].nunique():,}",
    )

    col_ratings.metric(
        "Ratings",
        f"{len(df):,}",
    )

    st.dataframe(
        movies.sort_values("title").head(100),
        hide_index=True,
        use_container_width=True,
    )