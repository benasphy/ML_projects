import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run():
    st.header("Breast Cancer Prediction using SVM")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/SVM)", unsafe_allow_html=True)

    # Load dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Number of Samples:", len(df))
    with col2:
        target_dist = df['target'].value_counts()
        fig = px.pie(values=target_dist.values, 
                    names=['Benign', 'Malignant'],
                    title='Diagnosis Distribution')
        st.plotly_chart(fig)

    # Data Analysis
    st.subheader("Data Analysis")
    
    # Feature distributions
    st.write("Feature Distributions by Diagnosis")
    selected_feature = st.selectbox("Select Feature to View:", data.feature_names)
    
    fig = px.box(df, x='target', y=selected_feature,
                title=f'{selected_feature} Distribution by Diagnosis',
                labels={'target': 'Diagnosis', selected_feature: selected_feature})
    st.plotly_chart(fig)

    # PCA Visualization
    st.subheader("Data Visualization (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df.drop('target', axis=1))
    
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Diagnosis': ['Benign' if x == 1 else 'Malignant' for x in df['target']]
    })
    
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Diagnosis',
                    title='PCA Visualization of Breast Cancer Data')
    st.plotly_chart(fig)

    # Data preprocessing
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = SVC(kernel='rbf', probability=True)
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
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        st.metric("Cross-validation Score", f"{cv_scores.mean():.2%}")
    with col3:
        st.metric("Cross-validation Std", f"{cv_scores.std():.2%}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Malignant', 'Benign'],
                   y=['Malignant', 'Benign'],
                   text_auto=True,
                   aspect="auto")
    st.plotly_chart(fig)

    # Classification Report
    st.subheader("Detailed Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Feature Importance
    st.subheader("Feature Importance")
    # For SVM with RBF kernel, we'll use permutation importance
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
    
    feature_importance = pd.DataFrame({
        'Feature': data.feature_names,
        'Importance': result.importances_mean
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance.head(10), x='Feature', y='Importance',
                title='Top 10 Most Important Features')
    st.plotly_chart(fig)

    # Prediction interface
    st.subheader("Predict Breast Cancer")
    
    # Create input fields for each feature
    input_data = {}
    cols = st.columns(3)
    for i, feature in enumerate(data.feature_names):
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].mean())
            )

    if st.button("Predict"):
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Display prediction
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", "Benign" if prediction == 1 else "Malignant")
        with col2:
            st.metric("Confidence", f"{max(probabilities):.2%}")
        
        # Visualize prediction probabilities
        fig = go.Figure(data=[
            go.Bar(x=['Malignant', 'Benign'],
                  y=probabilities,
                  text=[f'{p:.2%}' for p in probabilities],
                  textposition='auto',
            )
        ])
        fig.update_layout(title='Prediction Probabilities',
                        xaxis_title='Diagnosis',
                        yaxis_title='Probability')
        st.plotly_chart(fig)
        
        # Feature Analysis
        st.subheader("Feature Analysis")
        
        # Compare input values with dataset statistics
        comparison_df = pd.DataFrame({
            'Feature': data.feature_names,
            'Your Value': input_data.values(),
            'Dataset Mean': df.drop('target', axis=1).mean(),
            'Dataset Std': df.drop('target', axis=1).std()
        })
        
        # Calculate z-scores
        comparison_df['Z-Score'] = (comparison_df['Your Value'] - comparison_df['Dataset Mean']) / comparison_df['Dataset Std']
        
        # Plot feature comparison
        fig = px.bar(comparison_df.head(10), x='Feature', y='Z-Score',
                    title='Feature Comparison (Z-Scores)',
                    color='Z-Score',
                    color_continuous_scale=['red', 'white', 'green'])
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig)

if __name__ == "__main__":
    run() 