import streamlit as st
from KNN_projects import (
    movie_recommendation,
    tshirt_size_prediction,
)

def run():
    st.title("KNN Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Movie Recommendation",
            "T-Shirt Size Prediction",
        ],
    )

    # Run the selected project
    if project == "Movie Recommendation":
        movie_recommendation.run()
    elif project == "T-Shirt Size Prediction":
        tshirt_size_prediction.run()

if __name__ == "__main__":
    run()
