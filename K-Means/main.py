import streamlit as st
import sys
from pathlib import Path

# Ensure the Decision_Trees_projects folder is in the Python path
sys.path.append(str(Path(__file__).parent))
from K_Means_projects import (
    loan_approval,
    customer_segmentation
)

def run():
    st.title("K-Means Clustering Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Loan Approval",
            "Customer Segmentation"
        ],
    )

    # Run the selected project
    if project == "Loan Approval":
        loan_approval.run()
    elif project == "Customer Segmentation":
        customer_segmentation.run()

if __name__ == "__main__":
    run()
