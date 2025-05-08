import streamlit as st
import importlib

st.set_page_config(page_title="ML Projects", layout="wide")

st.title("📊 Machine Learning Portfolio")
st.markdown("Explore various ML projects under Linear Regression.")

# Top-level category — you can later add others like Classification, Clustering, etc.
category = st.selectbox("Choose a Topic", ["Linear Regression"])

if category == "Linear Regression":
    subproject = st.radio("Choose a Linear Regression Project", [
        "Simple Linear Regression",
        "Multivariable Linear Regression",
        "Ridge and Lasso Regression"
    ])

    if subproject == "Simple Linear Regression":
        module = importlib.import_module("LinearRegression.simple_lr")
    elif subproject == "Multivariable Linear Regression":
        module = importlib.import_module("LinearRegression.multivariable_lr")
    elif subproject == "Ridge and Lasso Regression":
        module = importlib.import_module("LinearRegression.ridge_lasso")

    module.run()
