import streamlit as st
import sys
from pathlib import Path

# Ensure the SVM_projects folder is in the Python path
sys.path.append(str(Path(__file__).parent))

from SVM_projects import (
    spam_detection,
    breast_cancer_prediction,
)

def run():
    st.title("SVM Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Spam Detection",
            "Breast Cancer Prediction",
        ],
    )

    # Run the selected project
    if project == "Spam Detection":
        spam_detection.run()
    elif project == "Breast Cancer Prediction":
        breast_cancer_prediction.run()

if __name__ == "__main__":
    run()
