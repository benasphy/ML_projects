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

def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    age = np.random.normal(35, 10, n_samples)
    age = np.clip(age, 18, 65)
    
    risk_factors = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.6, 0.3, 0.1])
    sexual_activity = np.random.choice(['None', 'Protected', 'Unprotected'], n_samples, p=[0.3, 0.5, 0.2])
    drug_use = np.random.choice(['None', 'Past', 'Current'], n_samples, p=[0.7, 0.2, 0.1])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'RiskFactor': risk_factors,
        'SexualActivity': sexual_activity,
        'DrugUse': drug_use
    })
    
    # Generate target (HIV status) with some patterns
    base_prob = 0.05
    risk_effect = {
        'Low': 0.5,
        'Medium': 1.0,
        'High': 2.0
    }
    activity_effect = {
        'None': 0.3,
        'Protected': 0.7,
        'Unprotected': 1.5
    }
    drug_effect = {
        'None': 0.5,
        'Past': 1.2,
        'Current': 1.8
    }
    
    # Calculate probability of HIV
    prob = base_prob * \
           df['RiskFactor'].map(risk_effect) * \
           df['SexualActivity'].map(activity_effect) * \
           df['DrugUse'].map(drug_effect) * \
           (1 + (df['Age'] - 35) / 100)
    
    # Generate actual HIV status
    df['HIV'] = np.random.binomial(1, prob)
    
    return df

def run():
    st.header("HIV Risk Prediction using Logistic Regression")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/Logistic_Regression)", unsafe_allow_html=True)

    # Generate sample data
    df = generate_sample_data()
    
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Features:", ", ".join(df.columns[:-1]))
        st.write("Target: HIV Status (0: Negative, 1: Positive)")
    with col2:
        st.write("Class Distribution:")
        class_dist = df['HIV'].value_counts()
        fig = px.pie(values=class_dist.values, names=['Negative', 'Positive'], 
                    title='HIV Status Distribution')
        st.plotly_chart(fig)

    # Prepare data
    X = pd.get_dummies(df[['Age', 'RiskFactor', 'SexualActivity', 'DrugUse']])
    y = df['HIV']
    
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
                   x=['Negative', 'Positive'],
                   y=['Negative', 'Positive'],
                   text_auto=True,
                   aspect="auto")
    st.plotly_chart(fig)
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(model.coef_[0])
    })
    fig = px.bar(importance, x='Feature', y='Importance',
                title='Feature Importance in Prediction')
    st.plotly_chart(fig)
    
    # Interactive Prediction
    st.subheader("Make a Prediction")
    st.write("Enter patient information:")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 35)
        risk_factor = st.selectbox("Risk Factor", df['RiskFactor'].unique())
    with col2:
        sexual_activity = st.selectbox("Sexual Activity", df['SexualActivity'].unique())
        drug_use = st.selectbox("Drug Use", df['DrugUse'].unique())
    
    if st.button("Predict"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'RiskFactor': [risk_factor],
            'SexualActivity': [sexual_activity],
            'DrugUse': [drug_use]
        })
        
        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_data)
        # Ensure all columns from training data are present
        for col in X.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[X.columns]
        
        # Scale input data
        input_scaled = scaler.transform(input_encoded)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display prediction
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", "Positive" if prediction == 1 else "Negative")
        with col2:
            st.metric("Risk Probability", f"{probability[1]:.2%}")
        
        # Visualize prediction probability
        fig = go.Figure(data=[
            go.Bar(x=['Negative', 'Positive'],
                  y=probability,
                  text=[f'{p:.2%}' for p in probability],
                  textposition='auto',
            )
        ])
        fig.update_layout(title='Prediction Probabilities')
        st.plotly_chart(fig)
    
    # Data Visualization
    st.subheader("Data Analysis")
    
    # Age distribution by HIV status
    fig = px.histogram(df, x='Age', color='HIV',
                      title='Age Distribution by HIV Status',
                      barmode='overlay',
                      labels={'HIV': 'HIV Status'})
    st.plotly_chart(fig)
    
    # Risk factor analysis
    fig = px.box(df, x='RiskFactor', y='Age', color='HIV',
                title='Age Distribution by Risk Factor and HIV Status')
    st.plotly_chart(fig)
    
    # Sexual activity analysis
    fig = px.sunburst(df, path=['SexualActivity', 'HIV'],
                     title='HIV Status by Sexual Activity')
    st.plotly_chart(fig)
    
    # Drug use analysis
    fig = px.treemap(df, path=['DrugUse', 'HIV'],
                    title='HIV Status by Drug Use')
    st.plotly_chart(fig)

if __name__ == "__main__":
    run()