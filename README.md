# Movie Recommendation System

An interactive Streamlit app for movie recommendations using content-based filtering, collaborative filtering with SVD, and a hybrid recommender.

## Features

- Select a user ID and favorite movie to generate hybrid recommendations.
- Filter recommendations by preferred genres.
- View popular genre-based recommendations.
- Evaluate Content-Based, Collaborative SVD, and Hybrid models separately.
- Compare RMSE, MAE, Precision, Recall, and F1-Score using an 80/20 train/test split.

## Project Structure

```text
.
├── app.py
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── processed_movies.csv
├── requirements.txt
└── src/
    ├── collaborative.py
    ├── content_based.py
    ├── evaluation.py
    ├── hybrid_recommendation.py
    └── preprocessing.py
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Run the model evaluation from the command line:

```bash
python src/evaluation.py
```

## Deploy From GitHub

This is a Streamlit app, so it cannot be hosted as a normal static GitHub Pages site. Use Streamlit Community Cloud instead:

1. Push this project to a GitHub repository.
2. Go to https://share.streamlit.io/.
3. Create a new app from your GitHub repo.
4. Set the main file path to `app.py`.
5. Deploy.

## Dataset

The app expects these files to exist in the `data/` folder:

- `movies.csv`
- `ratings.csv`
- `processed_movies.csv`

You can regenerate `processed_movies.csv` with:

```bash
python src/preprocessing.py
```

## Notes

- GitHub accepts files up to 100 MB. The included CSV files are below that limit.
- If you deploy on Streamlit Community Cloud, make sure the `data/` folder is committed to the repository.
