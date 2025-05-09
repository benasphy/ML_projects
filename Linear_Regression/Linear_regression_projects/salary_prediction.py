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
    st.header("Salary Prediction using Linear Regression")
    st.markdown("[View this project on GitHub](../../Linear_Regression)", unsafe_allow_html=True)

    # Load dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Load default dataset
        st.info("Using default dataset: Salary_dataset.csv")
        df = pd.read_csv("Linear_Regression/Linear_Regression_projects/Salary_dataset.csv")

    # Remove 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Data Overview
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Number of records: {len(df)}")
        st.write("\n**Basic Statistics:**")
        st.write(df.describe().round(2))
    
    with col2:
        # Distribution plots
        fig = px.box(df, y=['YearsExperience', 'Salary'],
                    title='Years of Experience and Salary Distribution')
        st.plotly_chart(fig)

    # Data Analysis
    st.subheader("Data Analysis")
    
    # Correlation analysis
    correlation = df['YearsExperience'].corr(df['Salary'])
    st.write(f"**Correlation between Years of Experience and Salary:** {correlation:.3f}")
    
    # Scatter plot with trend line
    fig = px.scatter(df, x='YearsExperience', y='Salary',
                    title='Years of Experience vs Salary',
                    labels={'YearsExperience': 'Years of Experience',
                           'Salary': 'Salary ($)'})
    fig.add_trace(go.Scatter(x=df['YearsExperience'],
                            y=stats.linregress(df['YearsExperience'], df['Salary'])[0] * df['YearsExperience'] + 
                              stats.linregress(df['YearsExperience'], df['Salary'])[1],
                            mode='lines',
                            name='Trend Line'))
    st.plotly_chart(fig)

    # Split data
    X = df[['YearsExperience']]
    y = df['Salary']
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
        st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    with col3:
        st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")

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
                        labels={'x': 'Predicted Salary ($)',
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
    st.subheader("Salary Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        years_experience = st.number_input("Enter years of experience:",
                                         min_value=0.0,
                                         max_value=30.0,
                                         value=5.0,
                                         step=0.5)
        
        # Calculate prediction
        prediction = model.predict([[years_experience]])[0]
        confidence_interval = 1.96 * np.sqrt(mean_squared_error(y_test, y_pred))
        
        st.metric("Predicted Salary",
                 f"${prediction:,.2f}",
                 f"±${confidence_interval:,.2f}")
        
        # Add salary range information
        if prediction < 50000:
            st.warning("⚠️ The predicted salary is below the average market rate.")
            recommended_years = (50000 - model.intercept_) / model.coef_[0]
            st.info(f"To achieve an average salary ($50,000), consider gaining {recommended_years:.1f} years of experience.")
        elif prediction > 100000:
            st.success("🎯 The predicted salary is above the market average!")
    
    with col2:
        # Prediction visualization
        fig = go.Figure()
        
        # Add actual data points
        fig.add_trace(go.Scatter(
            x=df['YearsExperience'],
            y=df['Salary'],
            mode='markers',
            name='Actual Salaries',
            marker=dict(color='blue')
        ))
        
        # Add regression line
        x_range = np.linspace(df['YearsExperience'].min(), df['YearsExperience'].max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name='Regression Line',
            line=dict(color='red')
        ))
        
        # Add prediction point
        fig.add_trace(go.Scatter(
            x=[years_experience],
            y=[prediction],
            mode='markers',
            name='Prediction',
            marker=dict(color='green', size=12)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range + confidence_interval,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range - confidence_interval,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            name='95% Confidence Interval'
        ))
        
        # Add horizontal line at average salary
        avg_salary = df['Salary'].mean()
        fig.add_hline(y=avg_salary, line_dash="dash", line_color="orange", 
                     annotation_text="Average Salary", 
                     annotation_position="right")
        
        fig.update_layout(
            title='Years of Experience vs Salary Prediction',
            xaxis_title='Years of Experience',
            yaxis_title='Salary ($)',
            showlegend=True
        )
        st.plotly_chart(fig)

    # Model Information
    st.subheader("Model Information")
    st.write(f"**Salary increase per year of experience:** ${model.coef_[0]:,.2f}")
    st.write(f"**Starting salary (0 years):** ${model.intercept_:,.2f}")
    st.write(f"**Equation:** Salary = ${model.coef_[0]:,.2f} × Years + ${model.intercept_:,.2f}")

    # Career Insights
    st.subheader("Career Insights")
    
    # Calculate salary growth rate
    df['Salary_per_Year'] = df['Salary'] / df['YearsExperience']
    avg_growth_rate = df['Salary_per_Year'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Salary Growth Analysis:**")
        st.write(f"- Average salary increase per year: ${avg_growth_rate:,.2f}")
        st.write(f"- Each additional year of experience adds ${model.coef_[0]:,.2f} to salary")
        st.write(f"- Starting salary (0 years): ${model.intercept_:,.2f}")
    
    with col2:
        # Salary growth distribution
        fig = px.box(df, y='Salary_per_Year',
                    title='Salary Growth Rate Distribution',
                    labels={'Salary_per_Year': 'Salary per Year ($)'})
        st.plotly_chart(fig)

    # Career Milestones
    st.subheader("Career Milestones")
    
    # Calculate salary milestones
    milestones = [50000, 75000, 100000, 125000, 150000]
    st.write("**Expected years to reach salary milestones:**")
    for milestone in milestones:
        if milestone > model.intercept_:
            years = (milestone - model.intercept_) / model.coef_[0]
            st.write(f"- ${milestone:,.0f}: {years:.1f} years of experience")

if __name__ == "__main__":
    run()