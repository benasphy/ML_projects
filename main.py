import streamlit as st
import os

def run():
    st.title("Machine Learning Algorithms")

    # List of available algorithm folders
    algorithms = [
        "Linear Regression",
        "Logistic Regression",
        "Decision Trees",
        "Poisson Regression",
        "Support Vector Machines",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "GMM",
        "Hierarchical Clustering",
        "DBSCAN & HDBSCAN",
        "Fuzzy C-Means",
        "Association Rule Learning",
        "K-Means Clustering",
        "Dimensionality Reduction",
    ]

    # Sidebar to select an algorithm
    selected_algorithm = st.sidebar.selectbox("Select an Algorithm", algorithms)

    # Map algorithm names to folder paths
    algorithm_paths = {
        "Linear Regression": "Linear_Regression/main.py",
        "Logistic Regression": "Logistic_Regression/main.py",
        "Decision Trees": "Decision_Trees/main.py",
        "Poisson Regression": "Poisson_Regression/main.py",
        "Support Vector Machines": "SVM/main.py",
        "K-Nearest Neighbors": "KNN/main.py",
        "Naive Bayes": "Naive_Bayes/main.py",
        "GMM": "GMM/main.py",
        "Hierarchical Clustering": "Hierarchical_Clustering/main.py",
        "DBSCAN & HDBSCAN": "DBSCAN_HDBSCAN/main.py",
        "Fuzzy C-Means": "Fuzzy_C_Means/main.py",
        "Association Rule Learning": "Association_Rule_Learning/main.py",
        "K-Means Clustering": "K-Means/main.py",
        "Dimensionality Reduction": "Dimensionality_Reduction/main.py"
    }

    # Display instructions
    st.write(f"You selected: {selected_algorithm}")
    st.write("Click the button below to navigate to the selected algorithm's app.")

    # Button to navigate to the selected algorithm
    if st.button("Go to App"):
        selected_path = algorithm_paths[selected_algorithm]
        os.system(f"streamlit run {selected_path}")

if __name__ == "__main__":
    run()