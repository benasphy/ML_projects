import streamlit as st
import sys
from pathlib import Path

# Ensure the Logistic_Regression_projects folder is in the Python path
sys.path.append(str(Path(__file__).parent))

from Logistic_Regression_projects import (
    diabetes_prediction,
    rock_vs_mine,
    simple_hiv_prediction,
)

def run():
    st.title("Logistic Regression Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Diabetes Prediction",
            "Rock vs Mine",
            "Simple HIV Prediction",
        ],
    )

    # Run the selected project
    if project == "Diabetes Prediction":
        diabetes_prediction.run()
    elif project == "Rock vs Mine":
        rock_vs_mine.run()
    elif project == "Simple HIV Prediction":
        simple_hiv_prediction.run()

if __name__ == "__main__":
    run()