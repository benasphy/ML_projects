import streamlit as st
import sys
from pathlib import Path

# Ensure the Decision_Trees_projects folder is in the Python path
sys.path.append(str(Path(__file__).parent))
from Fuzzy_C_Means_projects import (
    image_segmentation,
    customer_profiling
)

def run():
    st.title("Fuzzy C-Means Clustering Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Image Segmentation",
            "Customer Profiling"
        ],
    )

    # Run the selected project
    if project == "Image Segmentation":
        image_segmentation.run()
    elif project == "Customer Profiling":
        customer_profiling.run()

if __name__ == "__main__":
    run() 