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
    traffic_volume = np.random.normal(5000, 2000, n_samples)
    traffic_volume = np.clip(traffic_volume, 1000, 10000)
    
    weather_conditions = np.random.choice(['Clear', 'Rainy', 'Snowy', 'Foggy'], n_samples, p=[0.6, 0.2, 0.1, 0.1])
    time_of_day = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples, p=[0.3, 0.3, 0.2, 0.2])
    road_type = np.random.choice(['Highway', 'Urban', 'Rural'], n_samples, p=[0.4, 0.4, 0.2])
    
    # Create DataFrame
    df = pd.DataFrame({
        'TrafficVolume': traffic_volume,
        'WeatherCondition': weather_conditions,
        'TimeOfDay': time_of_day,
        'RoadType': road_type
    })
    
    # Generate target (number of accidents) with some patterns
    base_rate = 0.001
    weather_effect = {
        'Clear': 1.0,
        'Rainy': 1.5,
        'Snowy': 2.0,
        'Foggy': 1.8
    }
    time_effect = {
        'Morning': 1.2,
        'Afternoon': 1.0,
        'Evening': 1.5,
        'Night': 1.8
    }
    road_effect = {
        'Highway': 1.0,
        'Urban': 1.5,
        'Rural': 0.8
    }
    
    # Calculate expected number of accidents
    expected_accidents = base_rate * \
                        df['TrafficVolume'] * \
                        df['WeatherCondition'].map(weather_effect) * \
                        df['TimeOfDay'].map(time_effect) * \
                        df['RoadType'].map(road_effect)
    
    # Generate actual number of accidents using Poisson distribution
    df['Accidents'] = np.random.poisson(expected_accidents)
    
    return df

def run():
    st.header("Car Accident Prediction using Poisson Regression")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/Poisson_Regression)", unsafe_allow_html=True)

    # Generate sample data
    df = generate_sample_data()
    
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Features:", ", ".join(df.columns[:-1]))
        st.write("Target: Number of Accidents")
    with col2:
        st.write("Accident Statistics:")
        st.write(f"Mean: {df['Accidents'].mean():.2f}")
        st.write(f"Max: {df['Accidents'].max()}")
        st.write(f"Min: {df['Accidents'].min()}")
    
    # Data distribution visualization
    fig = px.histogram(df, x='Accidents',
                      title='Distribution of Number of Accidents',
                      nbins=30)
    st.plotly_chart(fig)

    # Prepare data
    X = pd.get_dummies(df[['TrafficVolume', 'WeatherCondition', 'TimeOfDay', 'RoadType']])
    y = df['Accidents']
    
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
                    labels={'x': 'Actual Accidents', 'y': 'Predicted Accidents'},
                    title='Actual vs Predicted Accidents')
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
    st.write("Enter traffic conditions:")
    
    col1, col2 = st.columns(2)
    with col1:
        traffic_volume = st.slider("Traffic Volume", 1000, 10000, 5000)
        weather = st.selectbox("Weather Condition", df['WeatherCondition'].unique())
    with col2:
        time = st.selectbox("Time of Day", df['TimeOfDay'].unique())
        road = st.selectbox("Road Type", df['RoadType'].unique())
    
    if st.button("Predict"):
        # Prepare input data
        input_data = pd.DataFrame({
            'TrafficVolume': [traffic_volume],
            'WeatherCondition': [weather],
            'TimeOfDay': [time],
            'RoadType': [road]
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
        st.metric("Expected Number of Accidents", f"{prediction:.1f}")
        
        # Visualize prediction with confidence interval
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Predicted Accidents'],
            y=[prediction],
            error_y=dict(type='data', array=[np.sqrt(prediction)], visible=True),
            name='Prediction'
        ))
        fig.update_layout(title='Predicted Accidents with 95% Confidence Interval')
        st.plotly_chart(fig)
    
    # Data Analysis
    st.subheader("Data Analysis")
    
    # Weather impact
    fig = px.box(df, x='WeatherCondition', y='Accidents',
                title='Accident Distribution by Weather Condition')
    st.plotly_chart(fig)
    
    # Time of day impact
    fig = px.box(df, x='TimeOfDay', y='Accidents',
                title='Accident Distribution by Time of Day')
    st.plotly_chart(fig)
    
    # Road type impact
    fig = px.box(df, x='RoadType', y='Accidents',
                title='Accident Distribution by Road Type')
    st.plotly_chart(fig)
    
    # Traffic volume vs accidents
    fig = px.scatter(df, x='TrafficVolume', y='Accidents',
                    color='WeatherCondition',
                    title='Traffic Volume vs Accidents by Weather Condition')
    st.plotly_chart(fig)

if __name__ == "__main__":
    run()