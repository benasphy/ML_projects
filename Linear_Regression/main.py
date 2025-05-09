import streamlit as st
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import project modules
try:
    from Linear_regression_projects.messi_goal_prediction import run as messi_goal_prediction
    from Linear_regression_projects.house_price_prediction import run as house_price_prediction
    from Linear_regression_projects.study_hours_exam_prediction import run as study_hours_exam_prediction
    from Linear_regression_projects.normal_equation_vs_gradient_descent import run as normal_equation_vs_gradient_descent
    from Linear_regression_projects.salary_prediction import run as salary_prediction
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.write("Please make sure all required files are in the correct locations.")

def run():
    st.title("Linear Regression Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Messi Goal Prediction",
            "House Price Prediction",
            "Study Hours and Exam Prediction",
            "Normal Equation vs Gradient Descent",
            "Salary Prediction",
        ],
    )

    # Run the selected project
    try:
        if project == "Messi Goal Prediction":
            messi_goal_prediction()
        elif project == "House Price Prediction":
            house_price_prediction()
        elif project == "Study Hours and Exam Prediction":
            study_hours_exam_prediction()
        elif project == "Normal Equation vs Gradient Descent":
            normal_equation_vs_gradient_descent()
        elif project == "Salary Prediction":
            salary_prediction()
    except Exception as e:
        st.error(f"Error running {project}: {str(e)}")
        st.write("Please check if all required files and dependencies are available.")

if __name__ == "__main__":
    run()