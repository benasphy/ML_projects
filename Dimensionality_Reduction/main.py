import streamlit as st
import sys
from pathlib import Path

# Ensure the Decision_Trees_projects folder is in the Python path
sys.path.append(str(Path(__file__).parent))
from Dimensionality_Reduction_projects import (
    image_compression,
    feature_selection
)

def run():
    st.title("Dimensionality Reduction Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Image Compression",
            "Feature Selection"
        ],
    )

    # Run the selected project
    if project == "Image Compression":
        image_compression.run()
    elif project == "Feature Selection":
        feature_selection.run()

if __name__ == "__main__":
    run() 