import streamlit as st
import sys
from pathlib import Path

# Ensure the Decision_Trees_projects folder is in the Python path
sys.path.append(str(Path(__file__).parent))

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