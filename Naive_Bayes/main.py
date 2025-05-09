import streamlit as st
from Naive_Bayes_projects import (
    weather_prediction,
    spam_detection_nb,
    fake_news_prediction,
)

def run():
    st.title("Naive Bayes Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Weather Prediction",
            "Spam Detection",
            "Fake News Prediction",
        ],
    )

    # Run the selected project
    if project == "Weather Prediction":
        weather_prediction.run()
    elif project == "Spam Detection":
        spam_detection_nb.run()
    elif project == "Fake News Prediction":
        fake_news_prediction.run()

if __name__ == "__main__":
    run()