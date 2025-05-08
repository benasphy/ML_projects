import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def run():
    st.header("📊 Multivariable Linear Regression")

    X = np.random.rand(100, 2) * 10
    y = 1.5 * X[:, 0] + 3.2 * X[:, 1] + np.random.randn(100) * 2

    model = LinearRegression()
    model.fit(X, y)

    st.write(f"**Coefficients**: {model.coef_}")
    st.write(f"**Intercept**: {model.intercept_:.2f}")
