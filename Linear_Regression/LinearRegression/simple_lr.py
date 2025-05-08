import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run():
    st.header("📈 Simple Linear Regression")
    st.write("This app demonstrates a simple linear regression using dummy data.")

    # Generate sample data
    X = np.random.rand(50, 1) * 10
    y = 2.5 * X + np.random.randn(50, 1) * 2

    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict(X)

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Data")
    ax.plot(X, pred, color='red', label="Regression Line")
    ax.set_title("Simple Linear Regression")
    ax.legend()
    st.pyplot(fig)

    st.write(f"**Slope**: {model.coef_[0][0]:.2f}")
    st.write(f"**Intercept**: {model.intercept_[0]:.2f}")
