import streamlit as st
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