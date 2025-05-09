import streamlit as st
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import project modules
try:
    from Logistic_Regression_projects.diabetes_prediction import run as diabetes_prediction
    from Logistic_Regression_projects.rock_vs_mine import run as rock_vs_mine
    from Logistic_Regression_projects.simple_hiv_prediction import run as simple_hiv_prediction
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.write("Please make sure all required files are in the correct locations.")

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
    try:
        if project == "Diabetes Prediction":
            diabetes_prediction()
        elif project == "Rock vs Mine":
            rock_vs_mine()
        elif project == "Simple HIV Prediction":
            simple_hiv_prediction()
    except Exception as e:
        st.error(f"Error running {project}: {str(e)}")
        st.write("Please check if all required files and dependencies are available.")

if __name__ == "__main__":
    run()