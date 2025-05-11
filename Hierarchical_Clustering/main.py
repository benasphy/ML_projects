import streamlit as st
import sys
from pathlib import Path

# Ensure the Decision_Trees_projects folder is in the Python path
sys.path.append(str(Path(__file__).parent))
from Hierarchical_projects import (
    document_clustering,
    market_basket_analysis,
    loan_approval,
    customer_segmentation
)

def run():
    st.title("Clustering Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Document Clustering",
            "Market Basket Analysis",
            "Loan Approval",
            "Customer Segmentation"
        ],
    )

    # Run the selected project
    if project == "Document Clustering":
        document_clustering.run()
    elif project == "Market Basket Analysis":
        market_basket_analysis.run()
    elif project == "Loan Approval":
        loan_approval.run()
    elif project == "Customer Segmentation":
        customer_segmentation.run()

if __name__ == "__main__":
    run() 