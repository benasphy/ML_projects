import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    energy_levels = np.random.choice(['High', 'Low'], n_samples, p=[0.4, 0.6])
    motivation_levels = np.random.choice(['Highly Motivated', 'Neutral', 'No Motivation'], 
                                       n_samples, p=[0.3, 0.4, 0.3])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Energy': energy_levels,
        'Motivation': motivation_levels
    })
    
    # Generate target (gym attendance) with some patterns
    def determine_gym_attendance(row):
        if row['Energy'] == 'High' and row['Motivation'] in ['Highly Motivated', 'Neutral']:
            return 1
        elif row['Energy'] == 'Low' and row['Motivation'] == 'No Motivation':
            return 0
        else:
            # Add some randomness for other combinations
            return np.random.choice([0, 1], p=[0.7, 0.3])
    
    df['Gym'] = df.apply(determine_gym_attendance, axis=1)
    
    return df

def run():
    st.header("Gym Attendance Prediction using Decision Trees")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Decision_Trees)", unsafe_allow_html=True)

    # Generate sample data
    df = generate_sample_data()
    
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Features:", ", ".join(df.columns[:-1]))
        st.write("Target: Gym Attendance (0: No, 1: Yes)")
    with col2:
        st.write("Class Distribution:")
        class_dist = df['Gym'].value_counts()
        fig = px.pie(values=class_dist.values, names=['No', 'Yes'], 
                    title='Gym Attendance Distribution')
        st.plotly_chart(fig)

    # Feature Analysis
    st.subheader("Feature Analysis")
    
    # Energy level impact
    fig = px.histogram(df, x='Energy', color='Gym',
                      title='Gym Attendance by Energy Level',
                      barmode='group')
    st.plotly_chart(fig)
    
    # Motivation level impact
    fig = px.histogram(df, x='Motivation', color='Gym',
                      title='Gym Attendance by Motivation Level',
                      barmode='group')
    st.plotly_chart(fig)

    # Prepare data
    X = pd.get_dummies(df[['Energy', 'Motivation']])
    y = df['Gym']
    
    # Train model
    model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Model evaluation
    st.subheader("Model Performance")
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{classification_report(y, y_pred, output_dict=True)['1']['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{classification_report(y, y_pred, output_dict=True)['1']['recall']:.2%}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['No', 'Yes'],
                   y=['No', 'Yes'],
                   text_auto=True,
                   aspect="auto")
    st.plotly_chart(fig)
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })
    fig = px.bar(importance, x='Feature', y='Importance',
                title='Feature Importance in Prediction')
    st.plotly_chart(fig)
    
    # Decision Tree Visualization
    st.subheader("Decision Tree Structure")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], 
             filled=True, rounded=True, fontsize=10)
    st.pyplot(fig)
    
    # Interactive Prediction
    st.subheader("Make a Prediction")
    st.write("Enter your current state:")
    
    col1, col2 = st.columns(2)
    with col1:
        energy = st.selectbox("Energy Level:", df['Energy'].unique())
    with col2:
        motivation = st.selectbox("Motivation Level:", df['Motivation'].unique())
    
    if st.button("Predict"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Energy': [energy],
            'Motivation': [motivation]
        })
        
        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_data)
        # Ensure all columns from training data are present
        for col in X.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[X.columns]
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0]
        
        # Display prediction
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", "Will go to the gym" if prediction == 1 else "Will not go to the gym")
        with col2:
            st.metric("Confidence", f"{max(probability):.2%}")
        
        # Visualize prediction probability
        fig = go.Figure(data=[
            go.Bar(x=['No', 'Yes'],
                  y=probability,
                  text=[f'{p:.2%}' for p in probability],
                  textposition='auto',
            )
        ])
        fig.update_layout(title='Prediction Probabilities')
        st.plotly_chart(fig)
    
    # Data Insights
    st.subheader("Data Insights")
    
    # Energy and Motivation combination analysis
    fig = px.sunburst(df, path=['Energy', 'Motivation', 'Gym'],
                     title='Gym Attendance by Energy and Motivation Levels')
    st.plotly_chart(fig)
    
    # Success rate by feature combinations
    success_rate = df.groupby(['Energy', 'Motivation'])['Gym'].mean().reset_index()
    fig = px.treemap(success_rate, path=['Energy', 'Motivation'],
                    values='Gym',
                    title='Success Rate by Feature Combinations')
    st.plotly_chart(fig)

if __name__ == "__main__":
    run()