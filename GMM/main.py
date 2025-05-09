import streamlit as st
from GMM_projects import (
    customer_segmentation,
    image_color_segmentation,
)

def run():
    st.title("GMM Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Customer Segmentation",
            "Image Color Segmentation",
        ],
    )

    # Run the selected project
    if project == "Customer Segmentation":
        customer_segmentation.run()
    elif project == "Image Color Segmentation":
        image_color_segmentation.run()

if __name__ == "__main__":
    run()
