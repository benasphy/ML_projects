import streamlit as st
import sys
from pathlib import Path

# Ensure the Decision_Trees_projects folder is in the Python path
sys.path.append(str(Path(__file__).parent))
from DBSCAN_HDBSCAN_projects import (
    anomaly_detection,
    customer_behavior_analysis
)

def run():
    st.title("DBSCAN & HDBSCAN Clustering Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Anomaly Detection",
            "Customer Behavior Analysis"
        ],
    )

    # Run the selected project
    if project == "Anomaly Detection":
        anomaly_detection.run()
    elif project == "Customer Behavior Analysis":
        customer_behavior_analysis.run()

if __name__ == "__main__":
    run() 