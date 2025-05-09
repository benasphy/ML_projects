import streamlit as st
from Poisson_Regression_projects import (
    competition_award,
    no_of_car_accident,
)

def run():
    st.title("Poisson Regression Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Competition Award Prediction",
            "Number of Car Accidents Prediction",
        ],
    )

    # Run the selected project
    if project == "Competition Award Prediction":
        competition_award.run()
    elif project == "Number of Car Accidents Prediction":
        no_of_car_accident.run()

if __name__ == "__main__":
    run()