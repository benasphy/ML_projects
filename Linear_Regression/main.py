import streamlit as st
from Linear_regression_projects import (
    messi_goal_prediction,
    house_price_prediction,
    study_hours_exam_prediction,
    normal_equation_vs_gradient_descent,
    salary_prediction,
)

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
if project == "Messi Goal Prediction":
    messi_goal_prediction.run()
elif project == "House Price Prediction":
    house_price_prediction.run()
elif project == "Study Hours and Exam Prediction":
    study_hours_exam_prediction.run()
elif project == "Normal Equation vs Gradient Descent":
    normal_equation_vs_gradient_descent.run()
elif project == "Salary Prediction":
    salary_prediction.run()