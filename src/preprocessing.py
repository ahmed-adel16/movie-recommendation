import pandas as pd


# =========================
# Load datasets
# =========================

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags=  pd.read_csv('data/tags.csv')

# =========================
# Missing values
# =========================

print("Movies Missing Values:")
print(movies.isnull().sum())

print("\nRatings Missing Values:")
print(ratings.isnull().sum())

print("\nTags Missing Values:")
print(tags.isnull().sum())

# =========================
# Remove duplicates
# =========================

movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)
tags.drop_duplicates(inplace=True)


# ========================
# Aggregate tags per movie
# ========================

movie_tags = tags.groupby("movieId")["tag"].apply(
    lambda x: " ".join(x.astype(str))
).reset_index()


# =========================
# Clean genres
# =========================

movies['genres'] = movies['genres'].str.replace(
    '|',
    ' ',
    regex=False
)


# =========================
# Merge datasets
# =========================

df = pd.merge(
    ratings,
    movies,
    on='movieId'
)

df = pd.merge(
    df,
    movie_tags,
    on='movieId'
)

# =========================
# Display samples
# =========================

print("\nMerged Dataset:")
print(df.head())

df["combined_features"] = (
    df["genres"] + " " +
    df["tag"]
)

# =========================
# Check rating range (to check outliers)
# =========================
print(f"rating range: {df['rating'].min()} - {df['rating'].max()}")

# =========================
# Save processed dataset
# =========================

df.to_csv(
    "data/processed_movies.csv",
    index=False
)

print("\nProcessed dataset saved successfully.")

print("\nProcessed dataset columns:-\n", df.columns.tolist())