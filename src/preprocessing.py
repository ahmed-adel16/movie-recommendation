import pandas as pd


# =========================
# Load datasets
# =========================

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")


# =========================
# Missing values
# =========================

print("Movies Missing Values:")
print(movies.isnull().sum())

print("\nRatings Missing Values:")
print(ratings.isnull().sum())


# =========================
# Remove duplicates
# =========================

movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)


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


# =========================
# Display samples
# =========================

print("\nMerged Dataset:")
print(df.head())

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