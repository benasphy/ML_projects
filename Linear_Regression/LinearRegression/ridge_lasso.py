import streamlit as st
from sklearn.linear_model import Ridge, Lasso
import numpy as np

def run():
    st.header("📉 Ridge and Lasso Regression")

    X = np.random.rand(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1)

    ridge.fit(X, y)
    lasso.fit(X, y)

    st.subheader("🔵 Ridge Regression")
    st.write(f"Coefficients: {ridge.coef_}")
    st.write(f"Intercept: {ridge.intercept_:.2f}")

    st.subheader("🔴 Lasso Regression")
    st.write(f"Coefficients: {lasso.coef_}")
    st.write(f"Intercept: {lasso.intercept_:.2f}")
