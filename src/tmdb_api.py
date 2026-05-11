import pandas as pd
import requests
import streamlit as st

API_KEY = 'cc141634c00faa7070493f6a051c406c'

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

links_df = pd.read_csv("data/links.csv")

links_df["tmdbId"] = pd.to_numeric(links_df["tmdbId"], errors="coerce")


@st.cache_data
def get_tmdb_id(movie_id):

    row = links_df[links_df["movieId"] == movie_id]

    if row.empty:
        return None

    tmdb_id = row.iloc[0]["tmdbId"]

    if pd.isna(tmdb_id):
        return None

    return int(tmdb_id)


@st.cache_data
def fetch_movie_details(movie_id):

    tmdb_id = get_tmdb_id(movie_id)

    if tmdb_id is None:
        return None

    url = f"{BASE_URL}/movie/{tmdb_id}"

    response = requests.get(
        url,
        params={"api_key": API_KEY},
    )

    if response.status_code != 200:
        return None

    movie = response.json()

    poster_path = movie.get("poster_path")

    return {
        "title": movie.get("title"),
        "overview": movie.get("overview"),
        "poster": (
            IMAGE_BASE + poster_path
            if poster_path
            else None
        ),
        "rating": movie.get("vote_average"),
    }