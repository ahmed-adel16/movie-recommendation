# src/ui_components.py

import streamlit as st


def movie_card(movie):

    with st.container(border=True):

        st.image(
            movie["poster"],
            use_container_width=True
        )

        st.subheader(movie["title"])

        st.caption(
            f"⭐ {movie['score']:.2f}"
        )

        st.write(movie["genres"])

        st.write(movie["tags"])

        if movie.get("overview"):
            st.write(
                movie["overview"][:150] + "..."
            )