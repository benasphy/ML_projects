import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    experience = np.random.normal(5, 2, n_samples)
    experience = np.clip(experience, 0, 10)
    
    skill_level = np.random.choice(['Beginner', 'Intermediate', 'Advanced', 'Expert'], 
                                 n_samples, p=[0.3, 0.4, 0.2, 0.1])
    competition_type = np.random.choice(['Local', 'Regional', 'National', 'International'], 
                                      n_samples, p=[0.4, 0.3, 0.2, 0.1])
    preparation_hours = np.random.normal(20, 10, n_samples)
    preparation_hours = np.clip(preparation_hours, 5, 40)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Experience': experience,
        'SkillLevel': skill_level,
        'CompetitionType': competition_type,
        'PreparationHours': preparation_hours
    })
    
    # Generate target (number of awards) with some patterns
    base_rate = 0.5
    skill_effect = {
        'Beginner': 0.5,
        'Intermediate': 1.0,
        'Advanced': 1.5,
        'Expert': 2.0
    }
    competition_effect = {
        'Local': 0.8,
        'Regional': 1.0,
        'National': 1.3,
        'International': 1.5
    }
    
    # Calculate expected number of awards
    expected_awards = base_rate * \
                     (1 + df['Experience']/10) * \
                     df['SkillLevel'].map(skill_effect) * \
                     df['CompetitionType'].map(competition_effect) * \
                     (1 + df['PreparationHours']/100)
    
    # Generate actual number of awards using Poisson distribution
    df['Awards'] = np.random.poisson(expected_awards)
    
    return df

def run():
    st.header("Competition Awards Prediction using Poisson Regression")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/Poisson_Regression)", unsafe_allow_html=True)

    # Generate sample data
    df = generate_sample_data()
    
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Features:", ", ".join(df.columns[:-1]))
        st.write("Target: Number of Awards")
    with col2:
        st.write("Award Statistics:")
        st.write(f"Mean: {df['Awards'].mean():.2f}")
        st.write(f"Max: {df['Awards'].max()}")
        st.write(f"Min: {df['Awards'].min()}")
    
    # Data distribution visualization
    fig = px.histogram(df, x='Awards',
                      title='Distribution of Number of Awards',
                      nbins=30)
    st.plotly_chart(fig)

    # Prepare data
    X = pd.get_dummies(df[['Experience', 'SkillLevel', 'CompetitionType', 'PreparationHours']])
    y = df['Awards']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = PoissonRegressor(alpha=0.1)
    model.fit(X_train_scaled, y_train)
    
    # Model evaluation
    st.subheader("Model Performance")
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.2f}")
    with col2:
        st.metric("R² Score", f"{r2:.2%}")
    
    # Actual vs Predicted Plot
    fig = px.scatter(x=y_test, y=y_pred,
                    labels={'x': 'Actual Awards', 'y': 'Predicted Awards'},
                    title='Actual vs Predicted Awards')
    fig.add_trace(go.Scatter(x=[0, max(y_test)], y=[0, max(y_test)],
                            mode='lines', name='Perfect Prediction'))
    st.plotly_chart(fig)
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(model.coef_)
    })
    fig = px.bar(importance, x='Feature', y='Importance',
                title='Feature Importance in Prediction')
    st.plotly_chart(fig)
    
    # Interactive Prediction
    st.subheader("Make a Prediction")
    st.write("Enter competitor information:")
    
    col1, col2 = st.columns(2)
    with col1:
        experience = st.slider("Years of Experience", 0, 10, 5)
        skill_level = st.selectbox("Skill Level", df['SkillLevel'].unique())
    with col2:
        competition_type = st.selectbox("Competition Type", df['CompetitionType'].unique())
        preparation_hours = st.slider("Preparation Hours", 5, 40, 20)
    
    if st.button("Predict"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Experience': [experience],
            'SkillLevel': [skill_level],
            'CompetitionType': [competition_type],
            'PreparationHours': [preparation_hours]
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
        
        # Display prediction
        st.subheader("Prediction Result")
        st.metric("Expected Number of Awards", f"{prediction:.1f}")
        
        # Visualize prediction with confidence interval
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Predicted Awards'],
            y=[prediction],
            error_y=dict(type='data', array=[np.sqrt(prediction)], visible=True),
            name='Prediction'
        ))
        fig.update_layout(title='Predicted Awards with 95% Confidence Interval')
        st.plotly_chart(fig)
    
    # Data Analysis
    st.subheader("Data Analysis")
    
    # Skill level impact
    fig = px.box(df, x='SkillLevel', y='Awards',
                title='Award Distribution by Skill Level')
    st.plotly_chart(fig)
    
    # Competition type impact
    fig = px.box(df, x='CompetitionType', y='Awards',
                title='Award Distribution by Competition Type')
    st.plotly_chart(fig)
    
    # Experience vs Awards
    fig = px.scatter(df, x='Experience', y='Awards',
                    color='SkillLevel',
                    title='Experience vs Awards by Skill Level')
    st.plotly_chart(fig)
    
    # Preparation hours impact
    fig = px.scatter(df, x='PreparationHours', y='Awards',
                    color='CompetitionType',
                    title='Preparation Hours vs Awards by Competition Type')
    st.plotly_chart(fig)

if __name__ == "__main__":
    run()