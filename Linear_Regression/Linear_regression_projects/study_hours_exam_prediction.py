import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def calculate_residuals(y_true, y_pred):
    """Calculate and return residuals."""
    return y_true - y_pred

def run():
    st.header("Study Hours vs Exam Score Prediction")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Linear_Regression)", unsafe_allow_html=True)

    # Example data
    study_hours = np.array([2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4])
    exam_scores = np.array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69])

    # Create DataFrame
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Exam_Score': exam_scores
    })

    # Data Overview
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Number of students: {len(df)}")
        st.write("\n**Basic Statistics:**")
        st.write(df.describe().round(2))
    
    with col2:
        # Distribution plots
        fig = px.box(df, y=['Study_Hours', 'Exam_Score'],
                    title='Study Hours and Exam Scores Distribution')
        st.plotly_chart(fig)

    # Data Analysis
    st.subheader("Data Analysis")
    
    # Correlation analysis
    correlation = df['Study_Hours'].corr(df['Exam_Score'])
    st.write(f"**Correlation between Study Hours and Exam Score:** {correlation:.3f}")
    
    # Scatter plot with trend line
    fig = px.scatter(df, x='Study_Hours', y='Exam_Score',
                    title='Study Hours vs Exam Score',
                    labels={'Study_Hours': 'Study Hours',
                           'Exam_Score': 'Exam Score'})
    fig.add_trace(go.Scatter(x=df['Study_Hours'],
                            y=stats.linregress(df['Study_Hours'], df['Exam_Score'])[0] * df['Study_Hours'] + 
                              stats.linregress(df['Study_Hours'], df['Exam_Score'])[1],
                            mode='lines',
                            name='Trend Line'))
    st.plotly_chart(fig)

    # Split data
    X = df[['Study_Hours']]
    y = df['Exam_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model Evaluation
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
    with col2:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f} points")
    with col3:
        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f} points")

    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5)
    st.write(f"**Cross-validation R² scores:** {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

    # Residual Analysis
    st.subheader("Residual Analysis")
    residuals = calculate_residuals(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        # Residuals vs Predicted
        fig = px.scatter(x=y_pred, y=residuals,
                        title='Residuals vs Predicted Values',
                        labels={'x': 'Predicted Score',
                               'y': 'Residuals'})
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig)
    
    with col2:
        # Residuals distribution
        fig = px.histogram(residuals,
                          title='Residuals Distribution',
                          labels={'value': 'Residuals'})
        st.plotly_chart(fig)

    # Prediction Interface
    st.subheader("Score Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.number_input("Enter study hours:",
                                    min_value=0.0,
                                    max_value=24.0,
                                    value=5.0,
                                    step=0.5)
        
        # Calculate prediction
        prediction = model.predict([[study_hours]])[0]
        confidence_interval = 1.96 * np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Handle scores above 100
        if prediction > 100:
            st.warning("⚠️ The predicted score exceeds 100 points. This might not be realistic for a standard exam.")
            prediction = min(prediction, 100)  # Cap the prediction at 100
            st.metric("Predicted Score (Capped)",
                     f"{prediction:.1f}",
                     f"±{confidence_interval:.1f}")
            st.info("Note: The original prediction was {:.1f} points, but has been capped at 100.".format(
                model.predict([[study_hours]])[0]))
        else:
            st.metric("Predicted Score",
                     f"{prediction:.1f}",
                     f"±{confidence_interval:.1f}")
        
        # Add study time recommendations
        if prediction < 60:
            st.warning("⚠️ The predicted score is below passing grade (60). Consider increasing study time.")
            recommended_hours = (60 - model.intercept_) / model.coef_[0]
            st.info(f"To achieve a passing score (60), consider studying for {recommended_hours:.1f} hours.")
        elif prediction > 90:
            st.success("🎯 The predicted score is excellent! You're on track for a high grade.")
    
    with col2:
        # Prediction visualization
        fig = go.Figure()
        
        # Add actual data points
        fig.add_trace(go.Scatter(
            x=df['Study_Hours'],
            y=df['Exam_Score'],
            mode='markers',
            name='Actual Scores',
            marker=dict(color='blue')
        ))
        
        # Add regression line
        x_range = np.linspace(df['Study_Hours'].min(), df['Study_Hours'].max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))
        # Cap the regression line at 100
        y_range = np.minimum(y_range, 100)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name='Regression Line',
            line=dict(color='red')
        ))
        
        # Add prediction point
        fig.add_trace(go.Scatter(
            x=[study_hours],
            y=[prediction],
            mode='markers',
            name='Prediction',
            marker=dict(color='green', size=12)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=x_range,
            y=np.minimum(y_range + confidence_interval, 100),
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_range,
            y=np.maximum(y_range - confidence_interval, 0),
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            name='95% Confidence Interval'
        ))
        
        # Add horizontal line at 100
        fig.add_hline(y=100, line_dash="dash", line_color="red", 
                     annotation_text="Maximum Score", 
                     annotation_position="right")
        
        # Add horizontal line at 60
        fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                     annotation_text="Passing Grade", 
                     annotation_position="right")
        
        fig.update_layout(
            title='Study Hours vs Exam Score Prediction',
            xaxis_title='Study Hours',
            yaxis_title='Exam Score',
            yaxis_range=[0, 105],  # Set y-axis range to show the 100 mark
            showlegend=True
        )
        st.plotly_chart(fig)

    # Model Information
    st.subheader("Model Information")
    st.write(f"**Slope (Score per hour):** {model.coef_[0]:.2f}")
    st.write(f"**Intercept:** {model.intercept_:.2f}")
    st.write(f"**Equation:** Score = {model.coef_[0]:.2f} × Hours + {model.intercept_:.2f}")

    # Study Insights
    st.subheader("Study Insights")
    
    # Calculate study efficiency
    df['Score_per_Hour'] = df['Exam_Score'] / df['Study_Hours']
    avg_efficiency = df['Score_per_Hour'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Study Efficiency Analysis:**")
        st.write(f"- Average points per hour: {avg_efficiency:.2f}")
        st.write(f"- Each additional hour of study is associated with {model.coef_[0]:.2f} points")
        st.write(f"- Base score (0 hours): {model.intercept_:.2f}")
    
    with col2:
        # Study efficiency distribution
        fig = px.box(df, y='Score_per_Hour',
                    title='Study Efficiency Distribution',
                    labels={'Score_per_Hour': 'Points per Hour'})
        st.plotly_chart(fig)

    # Study Recommendations
    st.subheader("Study Recommendations")
    
    # Calculate optimal study hours
    optimal_hours = df.loc[df['Score_per_Hour'].idxmax(), 'Study_Hours']
    max_efficiency = df['Score_per_Hour'].max()
    
    st.write("**Based on the data analysis:**")
    st.write(f"- Most efficient study duration: {optimal_hours:.1f} hours")
    st.write(f"- Maximum efficiency: {max_efficiency:.2f} points per hour")
    
    # Study hour ranges
    ranges = {
        'Light Study (1-3 hours)': df[(df['Study_Hours'] >= 1) & (df['Study_Hours'] <= 3)],
        'Moderate Study (4-6 hours)': df[(df['Study_Hours'] > 3) & (df['Study_Hours'] <= 6)],
        'Intensive Study (7+ hours)': df[df['Study_Hours'] > 6]
    }
    
    # Calculate statistics for each range
    range_stats = pd.DataFrame({
        'Study Range': list(ranges.keys()),
        'Average Score': [range_df['Exam_Score'].mean() for range_df in ranges.values()],
        'Average Hours': [range_df['Study_Hours'].mean() for range_df in ranges.values()],
        'Efficiency': [range_df['Score_per_Hour'].mean() for range_df in ranges.values()]
    })
    
    st.write("\n**Study Range Analysis:**")
    st.write(range_stats)
    
    # Plot study ranges
    fig = px.bar(range_stats, x='Study Range', y='Average Score',
                 title='Average Score by Study Range',
                 labels={'Study Range': 'Study Duration',
                        'Average Score': 'Average Exam Score'})
    st.plotly_chart(fig)

if __name__ == "__main__":
    run()