import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.header("T-Shirt Size Prediction using KNN")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/KNN)", unsafe_allow_html=True)

    # Load dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using default dataset: TShirt_size.csv")
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "TShirt_size.csv"))

    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Number of Samples:", len(df))
    with col2:
        size_dist = df["T Shirt Size"].value_counts()
        fig = px.pie(values=size_dist.values, names=size_dist.index,
                    title='T-Shirt Size Distribution')
        st.plotly_chart(fig)

    # Data Analysis
    st.subheader("Data Analysis")
    
    # Height and Weight Distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="T Shirt Size", y="Height (in cms)",
                    title='Height Distribution by Size')
        st.plotly_chart(fig)
    
    with col2:
        fig = px.box(df, x="T Shirt Size", y="Weight (in kgs)",
                    title='Weight Distribution by Size')
        st.plotly_chart(fig)

    # Scatter plot with size distribution
    fig = px.scatter(df, x="Height (in cms)", y="Weight (in kgs)",
                    color="T Shirt Size",
                    title='Height vs Weight by T-Shirt Size')
    st.plotly_chart(fig)

    # Data preprocessing
    encoder = LabelEncoder()
    df["T Shirt Size"] = encoder.fit_transform(df["T Shirt Size"])
    
    X = df[["Height (in cms)", "Weight (in kgs)"]]
    y = df["T Shirt Size"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
    model.fit(X_train, y_train)

    # Model evaluation
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        scores = cross_val_score(model, X, y, cv=5, scoring='precision_weighted')
        st.metric("Average Precision", f"{scores.mean():.2%}")
    with col2:
        st.metric("Standard Deviation", f"{scores.std():.2%}")
    with col3:
        accuracy = (y_pred == y_test).mean()
        st.metric("Test Accuracy", f"{accuracy:.2%}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Medium', 'Large'],
                   y=['Medium', 'Large'],
                   text_auto=True,
                   aspect="auto")
    st.plotly_chart(fig)

    # KNN Visualization
    st.subheader("KNN Decision Boundaries")
    
    # Create a mesh grid
    h_min, h_max = X["Height (in cms)"].min() - 1, X["Height (in cms)"].max() + 1
    w_min, w_max = X["Weight (in kgs)"].min() - 1, X["Weight (in kgs)"].max() + 1
    h_grid = np.arange(h_min, h_max, 0.5)
    w_grid = np.arange(w_min, w_max, 0.5)
    hh, ww = np.meshgrid(h_grid, w_grid)
    
    # Predict for each point in the grid
    grid_points = np.c_[hh.ravel(), ww.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    grid_predictions = model.predict(grid_points_scaled)
    
    # Plot decision boundaries
    fig = px.scatter(x=grid_points[:, 0], y=grid_points[:, 1],
                    color=grid_predictions,
                    title='KNN Decision Boundaries')
    fig.add_scatter(x=X["Height (in cms)"], y=X["Weight (in kgs)"],
                   mode='markers',
                   marker=dict(color=y, symbol='circle'),
                   name='Training Data')
    st.plotly_chart(fig)

    # Prediction interface
    st.subheader("Predict T-Shirt Size")
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("Height (in cms):", min_value=140, max_value=200, value=170)
    with col2:
        weight = st.number_input("Weight (in kgs):", min_value=40, max_value=120, value=70)

    if st.button("Predict Size"):
        new_sample = np.array([height, weight]).reshape(1, -1)
        new_sample_scaled = scaler.transform(new_sample)
        prediction = model.predict(new_sample_scaled)[0]
        probabilities = model.predict_proba(new_sample_scaled)[0]
        
        size_mapping = {0: "Large", 1: "Medium"}
        predicted_size = size_mapping[prediction]
        
        # Display prediction
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Size", predicted_size)
        with col2:
            st.metric("Confidence", f"{max(probabilities):.2%}")
        
        # Visualize prediction probabilities
        fig = go.Figure(data=[
            go.Bar(x=['Large', 'Medium'],
                  y=probabilities,
                  text=[f'{p:.2%}' for p in probabilities],
                  textposition='auto',
            )
        ])
        fig.update_layout(title='Prediction Probabilities',
                        xaxis_title='Size',
                        yaxis_title='Probability')
        st.plotly_chart(fig)
        
        # Show nearest neighbors
        st.subheader("Nearest Neighbors")
        distances, indices = model.kneighbors(new_sample_scaled)
        
        neighbors_df = pd.DataFrame({
            'Height (cm)': X.iloc[indices[0]]["Height (in cms)"],
            'Weight (kg)': X.iloc[indices[0]]["Weight (in kgs)"],
            'Size': [size_mapping[y.iloc[i]] for i in indices[0]],
            'Distance': distances[0]
        })
        
        fig = px.scatter(neighbors_df, x='Height (cm)', y='Weight (kg)',
                        color='Size',
                        size='Distance',
                        title='Nearest Neighbors',
                        hover_data=['Distance'])
        fig.add_scatter(x=[height], y=[weight],
                       mode='markers',
                       marker=dict(color='red', symbol='star', size=15),
                       name='Your Measurements')
        st.plotly_chart(fig)

if __name__ == "__main__":
    run() 