import streamlit as st
from Association_Rule_Learning_projects import (
    market_basket_analysis,
    recommendation_system
)

def run():
    st.title("Association Rule Learning Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Market Basket Analysis",
            "Recommendation System"
        ],
    )

    # Run the selected project
    if project == "Market Basket Analysis":
        market_basket_analysis.run()
    elif project == "Recommendation System":
        recommendation_system.run()

if __name__ == "__main__":
    run() 