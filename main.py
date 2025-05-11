import streamlit as st
import importlib

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

    # Map algorithm names to module paths
    algorithm_modules = {
        "Linear Regression": "Linear_Regression.main",
        "Logistic Regression": "Logistic_Regression.main",
        "Decision Trees": "Decision_Trees.main",
        "Poisson Regression": "Poisson_Regression.main",
        "Support Vector Machines": "SVM.main",
        "K-Nearest Neighbors": "KNN.main",
        "Naive Bayes": "Naive_Bayes.main",
        "GMM": "GMM.main",
        "Hierarchical Clustering": "Hierarchical_Clustering.main",
        "DBSCAN & HDBSCAN": "DBSCAN_HDBSCAN.main",
        "Fuzzy C-Means": "Fuzzy_C_Means.main",
        "Association Rule Learning": "Association_Rule_Learning.main",
        "K-Means Clustering": "K-Means.main",
        "Dimensionality Reduction": "Dimensionality_Reduction.main"
    }

    st.write(f"You selected: {selected_algorithm}")
    st.write("The selected algorithm's app will appear below.")

    # Dynamically import and run the selected module's run() function
    module_path = algorithm_modules[selected_algorithm]
    try:
        module = importlib.import_module(module_path)
        module.run()
    except Exception as e:
        st.error(f"Failed to load {selected_algorithm} app: {e}")

if __name__ == "__main__":
    run()