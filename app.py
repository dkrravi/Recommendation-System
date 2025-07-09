import streamlit as st
st.set_page_config(page_title="Movie Recommender", layout="centered")

import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data_and_model():
    df = pd.read_csv("movies_cleaned.csv")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    return df, tfidf_matrix

df, tfidf_matrix = load_data_and_model()
df = df.reset_index()
indices = pd.Series(df.index, index=df['movie_title'].str.strip().str.lower()).drop_duplicates()

def recommend_movies(title, tfidf_matrix=tfidf_matrix):
    title = title.lower().strip()
    if title not in indices:
        return ["Movie not found in the dataset."]
    idx = indices[title]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    seen_titles = set()
    recommendations = []
    for i in sim_scores:
        movie = df['movie_title'].iloc[i[0]].strip().title()
        if movie not in seen_titles:
            seen_titles.add(movie)
            recommendations.append(movie)
        if len(recommendations) == 5:
            break
    return recommendations

st.title("Movie Recommendation System")
st.write("Enter a movie title to see similar recommendations.")

movie_input = st.text_input("Enter movie title")

if movie_input:
    results = recommend_movies(movie_input)
    if results[0] == "Movie not found in the dataset.":
        st.error(results[0])
    else:
        st.write(f"Top 5 movies similar to '{movie_input.title()}':")
        for i, movie in enumerate(results, 1):
            st.write(f"{i}. {movie}")
