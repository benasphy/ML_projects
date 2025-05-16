import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def find_movie_matches(df, search_term):
    # Convert search term to lowercase for case-insensitive matching
    search_term = search_term.lower()
    # Find movies that contain the search term
    matches = df[df['title'].str.lower().str.contains(search_term, na=False)]
    return matches

def generate_wordcloud(texts, title):
    """Generate and display a word cloud."""
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         min_font_size=10).generate(' '.join(texts))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud)
    ax.axis('off')
    ax.set_title(title)
    return fig

def get_recommendations(df, movie_title, model, tfidf_matrix):
    # Get the index of the selected movie
    movie_idx = df[df['title'] == movie_title].index[0]
    
    # Find similar movies
    distances, indices = model.kneighbors(tfidf_matrix[movie_idx])
    
    # Create recommendations dataframe
    recommendations = pd.DataFrame({
        'Title': df.iloc[indices[0][1:]]['title'],
        'Type': df.iloc[indices[0][1:]]['type'],
        'Description': df.iloc[indices[0][1:]]['description'],
        'Similarity Score': 1 - distances[0][1:],  # Convert distance to similarity
        'Genre': df.iloc[indices[0][1:]]['listed_in'],
        'Director': df.iloc[indices[0][1:]]['director']
    })
    
    # Display selected movie info
    st.subheader("Selected Movie")
    selected_movie = df.iloc[movie_idx]
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Title:** {selected_movie['title']}")
        st.write(f"**Type:** {selected_movie['type']}")
        st.write(f"**Director:** {selected_movie['director']}")
    with col2:
        st.write(f"**Genre:** {selected_movie['listed_in']}")
        st.write(f"**Description:** {selected_movie['description']}")
    
    # Display recommendations
    st.subheader("Recommended Movies")
    
    # Sort recommendations by similarity score
    recommendations = recommendations.sort_values('Similarity Score', ascending=False)
    
    # Display each recommendation in an expander
    for idx, row in recommendations.iterrows():
        with st.expander(f"{row['Title']} (Similarity: {row['Similarity Score']:.2%})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Type:** {row['Type']}")
                st.write(f"**Director:** {row['Director']}")
            with col2:
                st.write(f"**Genre:** {row['Genre']}")
                st.write(f"**Description:** {row['Description']}")
    
    # Visualize similarity scores
    fig = px.bar(recommendations, x='Title', y='Similarity Score',
                title='Similarity Scores of Recommended Movies',
                labels={'Title': 'Movie Title', 'Similarity Score': 'Similarity Score'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    
    # Genre distribution of recommendations
    genre_counts = Counter()
    for genres in recommendations['Genre']:
        genre_counts.update([g.strip() for g in genres.split(',')])
    
    genre_df = pd.DataFrame({
        'Genre': list(genre_counts.keys()),
        'Count': list(genre_counts.values())
    }).sort_values('Count', ascending=False)
    
    fig = px.pie(genre_df, values='Count', names='Genre',
                title='Genre Distribution of Recommendations')
    st.plotly_chart(fig)

def run():
    st.header("Movie Recommendation System using KNN")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/KNN)", unsafe_allow_html=True)

    # Initialize session state for selected movie if it doesn't exist
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None

    # Load dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using default dataset: netflix_titles.csv")
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv("KNN/KNN_projects/netflix_titles.csv", encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            st.error(f"Error loading the dataset: {str(e)}")
            return

    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Number of Titles:", len(df))
    with col2:
        type_dist = df['type'].value_counts()
        fig = px.pie(values=type_dist.values, names=type_dist.index,
                    title='Content Type Distribution')
        st.plotly_chart(fig)

    # Data Analysis
    st.subheader("Data Analysis")
    
    # Genre distribution
    genre_counts = Counter()
    for genres in df['listed_in'].dropna():
        genre_counts.update([g.strip() for g in genres.split(',')])
    
    genre_df = pd.DataFrame({
        'Genre': list(genre_counts.keys()),
        'Count': list(genre_counts.values())
    }).sort_values('Count', ascending=False).head(10)
    
    fig = px.bar(genre_df, x='Genre', y='Count',
                title='Top 10 Genres')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    
    # Word cloud of descriptions
    st.write("Common Words in Movie Descriptions")
    descriptions = df['description'].dropna().tolist()
    fig = generate_wordcloud(descriptions, 'Movie Descriptions Word Cloud')
    st.pyplot(fig)

    # Data preprocessing
    df['description'] = df['description'].fillna('')
    df['combined_features'] = df['description'] + ' ' + df['listed_in'] + ' ' + df['director'].fillna('')

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Create KNN model
    model = NearestNeighbors(n_neighbors=6, metric='cosine')
    model.fit(tfidf_matrix)

    # Movie selection interface
    st.subheader("Find Similar Movies")
    
    # Add search functionality
    search_term = st.text_input("Search for a movie:", "")
    
    if search_term:
        matches = find_movie_matches(df, search_term)
        if not matches.empty:
            st.write("Found these movies:")
            for idx, row in matches.head(5).iterrows():
                if st.button(f"Select: {row['title']} ({row['type']})", key=f"search_{idx}"):
                    st.session_state.selected_movie = row['title']
    
    # Keep the dropdown for easy selection
    dropdown_movie = st.selectbox("Or select a movie from the list:", df['title'].tolist())
    
    # Show currently selected movie
    if st.session_state.selected_movie:
        st.write(f"Currently selected movie: {st.session_state.selected_movie}")
    
    # Recommendation button
    if st.button("Find Similar Movies"):
        # Use either the selected movie from search or the dropdown
        movie_to_use = st.session_state.selected_movie if st.session_state.selected_movie else dropdown_movie
        get_recommendations(df, movie_to_use, model, tfidf_matrix)
    
    # Add a button to clear selection
    if st.button("Clear Selection"):
        st.session_state.selected_movie = None
        st.rerun()

if __name__ == "__main__":
    run()
