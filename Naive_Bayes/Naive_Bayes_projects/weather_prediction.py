import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate weather features
    temperature = np.random.normal(25, 5, n_samples)  # Mean 25°C, std 5°C
    humidity = np.random.normal(60, 15, n_samples)    # Mean 60%, std 15%
    pressure = np.random.normal(1013, 5, n_samples)   # Mean 1013 hPa, std 5 hPa
    wind_speed = np.random.exponential(5, n_samples)  # Mean 5 m/s
    
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature': temperature,
        'Humidity': humidity,
        'Pressure': pressure,
        'Wind_Speed': wind_speed
    })
    
    # Generate weather conditions based on features
    def determine_weather(row):
        if row['Temperature'] > 30 and row['Humidity'] > 70:
            return 'Stormy'
        elif row['Temperature'] < 20 and row['Humidity'] > 80:
            return 'Rainy'
        elif row['Temperature'] > 25 and row['Humidity'] < 50:
            return 'Sunny'
        else:
            return 'Cloudy'
    
    df['Weather'] = df.apply(determine_weather, axis=1)
    
    return df

def run():
    st.header("Weather Prediction using Naive Bayes")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Naive_Bayes)", unsafe_allow_html=True)
    
    # Generate sample data
    df = generate_sample_data()
    
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Features:", ", ".join(df.columns[:-1]))
    with col2:
        st.write("Weather Distribution:")
        weather_dist = df['Weather'].value_counts()
        fig = px.pie(values=weather_dist.values, names=weather_dist.index,
                    title='Weather Conditions Distribution')
        st.plotly_chart(fig)
    
    # Feature Analysis
    st.subheader("Feature Analysis")
    
    # Temperature distribution by weather
    fig = px.box(df, x='Weather', y='Temperature',
                title='Temperature Distribution by Weather Condition')
    st.plotly_chart(fig)
    
    # Humidity distribution by weather
    fig = px.box(df, x='Weather', y='Humidity',
                title='Humidity Distribution by Weather Condition')
    st.plotly_chart(fig)
    
    # Feature correlations
    st.subheader("Feature Correlations")
    numeric_cols = ['Temperature', 'Humidity', 'Pressure', 'Wind_Speed']
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr,
                   labels=dict(x="Features", y="Features", color="Correlation"),
                   x=numeric_cols,
                   y=numeric_cols,
                   text_auto=True,
                   aspect="auto")
    st.plotly_chart(fig)
    
    # Prepare data for modeling
    X = df[['Temperature', 'Humidity', 'Pressure', 'Wind_Speed']]
    y = df['Weather']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Model evaluation
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Macro Precision", 
                 f"{classification_report(y_test, y_pred, output_dict=True)['macro avg']['precision']:.2%}")
    with col3:
        st.metric("Macro Recall",
                 f"{classification_report(y_test, y_pred, output_dict=True)['macro avg']['recall']:.2%}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=sorted(df['Weather'].unique()),
                   y=sorted(df['Weather'].unique()),
                   text_auto=True,
                   aspect="auto")
    st.plotly_chart(fig)
    
    # Feature Distributions
    st.subheader("Feature Distributions by Weather Condition")
    
    # Create subplot for feature distributions
    fig = plt.figure(figsize=(12, 8))
    for i, feature in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        for weather in df['Weather'].unique():
            sns.kdeplot(data=df[df['Weather'] == weather][feature], label=weather)
        plt.title(f'{feature} Distribution')
        plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Interactive Prediction
    st.subheader("Make a Weather Prediction")
    st.write("Enter weather conditions:")
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (°C)", float(df['Temperature'].min()), 
                              float(df['Temperature'].max()), 25.0)
        humidity = st.slider("Humidity (%)", float(df['Humidity'].min()), 
                           float(df['Humidity'].max()), 60.0)
    with col2:
        pressure = st.slider("Pressure (hPa)", float(df['Pressure'].min()), 
                           float(df['Pressure'].max()), 1013.0)
        wind_speed = st.slider("Wind Speed (m/s)", float(df['Wind_Speed'].min()), 
                             float(df['Wind_Speed'].max()), 5.0)
    
    if st.button("Predict Weather"):
        # Prepare input data
        input_data = np.array([[temperature, humidity, pressure, wind_speed]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Display prediction
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Weather", prediction)
        with col2:
            st.metric("Confidence", f"{max(probabilities):.2%}")
        
        # Visualize prediction probabilities
        fig = go.Figure(data=[
            go.Bar(x=sorted(df['Weather'].unique()),
                  y=probabilities,
                  text=[f'{p:.2%}' for p in probabilities],
                  textposition='auto',
            )
        ])
        fig.update_layout(title='Weather Condition Probabilities',
                        xaxis_title='Weather Condition',
                        yaxis_title='Probability')
        st.plotly_chart(fig)
        
        # Weather Condition Characteristics
        st.subheader("Typical Characteristics of Predicted Weather")
        weather_stats = df[df['Weather'] == prediction].describe()
        for feature in numeric_cols:
            st.write(f"**{feature}**:")
            st.write(f"- Average: {weather_stats[feature]['mean']:.2f}")
            st.write(f"- Range: {weather_stats[feature]['min']:.2f} to {weather_stats[feature]['max']:.2f}")

if __name__ == "__main__":
    run()