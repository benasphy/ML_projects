import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.header("Rock vs Mine Classification using Logistic Regression")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Logistic_Regression)", unsafe_allow_html=True)

    # Load dataset
    df = pd.read_csv("Logistic_Regression/Logistic_Regression_projects/Copy of sonar data.csv")
    
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Features: 60 frequency bands")
        st.write("Target: R (Rock) or M (Mine)")
    with col2:
        st.write("Class Distribution:")
        class_dist = df['R'].value_counts()
        fig = px.pie(values=class_dist.values, names=['Rock', 'Mine'], 
                    title='Rock vs Mine Distribution')
        st.plotly_chart(fig)

    # Feature selection
    st.subheader("Feature Selection")
    n_features = st.slider("Number of Features to Use", min_value=5, max_value=60, value=20)
    
    # Select top features based on variance
    feature_vars = df.iloc[:, :-1].var()
    selected_features = feature_vars.nlargest(n_features).index.tolist()

    # Prepare data
    X = df[selected_features]
    y = df['R'].map({'R': 0, 'M': 1})  # Convert to binary
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Model evaluation
    st.subheader("Model Performance")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{classification_report(y_test, y_pred, output_dict=True)['1']['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.2%}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Rock', 'Mine'],
                   y=['Rock', 'Mine'],
                   text_auto=True,
                   aspect="auto")
    st.plotly_chart(fig)
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': np.abs(model.coef_[0])
    })
    fig = px.bar(importance, x='Feature', y='Importance',
                title='Feature Importance in Prediction')
    st.plotly_chart(fig)
    
    # Interactive Prediction
    st.subheader("Make a Prediction")
    st.write("Enter frequency band values:")
    
    # Create input fields for selected features
    input_data = {}
    cols = st.columns(3)
    for i, feature in enumerate(selected_features):
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].mean())
            )
    
    if st.button("Predict"):
        # Scale input data
        input_scaled = scaler.transform(pd.DataFrame([input_data]))
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display prediction
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", "Mine" if prediction == 1 else "Rock")
        with col2:
            st.metric("Confidence", f"{max(probability):.2%}")
        
        # Visualize prediction probability
        fig = go.Figure(data=[
            go.Bar(x=['Rock', 'Mine'],
                  y=probability,
                  text=[f'{p:.2%}' for p in probability],
                  textposition='auto',
            )
        ])
        fig.update_layout(title='Prediction Probabilities')
        st.plotly_chart(fig)
    
    # Data Visualization
    st.subheader("Data Analysis")
    
    # PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot PCA results
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=df['R'].map({'R': 'Rock', 'M': 'Mine'}),
        title='PCA Visualization of Rock vs Mine Data',
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
    )
    st.plotly_chart(fig)
    
    # Feature correlation heatmap
    st.subheader("Feature Correlations")
    corr = df[selected_features].corr()
    fig = px.imshow(corr,
                   labels=dict(color="Correlation"),
                   x=corr.columns,
                   y=corr.columns,
                   aspect="auto")
    st.plotly_chart(fig)
    
    # Frequency band analysis
    st.subheader("Frequency Band Analysis")
    selected_band = st.selectbox("Select frequency band to analyze", selected_features)
    
    fig = px.box(df, x='R', y=selected_band,
                title=f'Distribution of {selected_band} by Class',
                labels={'R': 'Class', selected_band: 'Value'})
    st.plotly_chart(fig)

if __name__ == "__main__":
    run()