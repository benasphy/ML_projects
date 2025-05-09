import streamlit as st
from Decision_Trees_projects import (
    gym_decision_tree,
    gini_impurity_implementation,
)

def run():
    st.title("Decision Tree Projects")

    # Sidebar for project selection
    project = st.sidebar.selectbox(
        "Select a project",
        [
            "Gym Decision Tree",
            "Gini Impurity Implementation",
        ],
    )

    # Run the selected project
    if project == "Gym Decision Tree":
        gym_decision_tree.run()
    elif project == "Gini Impurity Implementation":
        gini_impurity_implementation.run()

if __name__ == "__main__":
    run()