import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def calculate_residuals(y_true, y_pred):
    """Calculate and return residuals."""
    return y_true - y_pred

def run():
    st.header("House Price Prediction")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/Linear_Regression)", unsafe_allow_html=True)

    # Example data
    house_sizes = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700])
    house_prices = np.array([245, 312, 279, 308, 199, 219, 405, 324, 319, 255])

    # Create DataFrame
    df = pd.DataFrame({
        'Size': house_sizes,
        'Price': house_prices
    })

    # Data Overview
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Number of houses: {len(df)}")
        st.write("\n**Basic Statistics:**")
        st.write(df.describe().round(2))
    
    with col2:
        # Distribution plots
        fig = px.box(df, title='Price and Size Distributions')
        st.plotly_chart(fig)

    # Data Analysis
    st.subheader("Data Analysis")
    
    # Correlation analysis
    correlation = df['Size'].corr(df['Price'])
    st.write(f"**Correlation between Size and Price:** {correlation:.3f}")
    
    # Scatter plot with trend line
    fig = px.scatter(df, x='Size', y='Price',
                    title='House Size vs Price',
                    labels={'Size': 'House Size (sq. ft.)',
                           'Price': 'Price ($1000)'})
    fig.add_trace(go.Scatter(x=df['Size'],
                            y=stats.linregress(df['Size'], df['Price'])[0] * df['Size'] + 
                              stats.linregress(df['Size'], df['Price'])[1],
                            mode='lines',
                            name='Trend Line'))
    st.plotly_chart(fig)

    # Reshape data
    X = house_sizes.reshape(-1, 1)
    y = house_prices

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Model Evaluation
    st.subheader("Model Evaluation")
    y_pred = model.predict(X)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2_score(y, y_pred):.3f}")
    with col2:
        st.metric("RMSE", f"${np.sqrt(mean_squared_error(y, y_pred)):.2f}K")
    with col3:
        st.metric("MAE", f"${mean_absolute_error(y, y_pred):.2f}K")

    # Residual Analysis
    st.subheader("Residual Analysis")
    residuals = calculate_residuals(y, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        # Residuals vs Predicted
        fig = px.scatter(x=y_pred, y=residuals,
                        title='Residuals vs Predicted Values',
                        labels={'x': 'Predicted Price ($1000)',
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
    st.subheader("Price Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        new_size = st.number_input("Enter house size (sq. ft.):",
                                 min_value=500,
                                 max_value=10000,
                                 value=2000,
                                 step=100)
        
        # Calculate prediction
        prediction = model.predict([[new_size]])[0]
        confidence_interval = 1.96 * np.sqrt(mean_squared_error(y, y_pred))
        
        st.metric("Predicted Price",
                 f"${prediction:.2f}K",
                 f"±${confidence_interval:.2f}K")
    
    with col2:
        # Prediction visualization
        fig = go.Figure()
        
        # Add actual data points
        fig.add_trace(go.Scatter(
            x=df['Size'],
            y=df['Price'],
            mode='markers',
            name='Actual Data',
            marker=dict(color='blue')
        ))
        
        # Add regression line
        x_range = np.linspace(min(df['Size']), max(df['Size']), 100)
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
            x=[new_size],
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
        
        fig.update_layout(
            title='House Price Prediction',
            xaxis_title='House Size (sq. ft.)',
            yaxis_title='Price ($1000)',
            showlegend=True
        )
        st.plotly_chart(fig)

    # Model Information
    st.subheader("Model Information")
    st.write(f"**Slope (Price per sq. ft.):** ${model.coef_[0]:.2f}")
    st.write(f"**Intercept:** ${model.intercept_:.2f}")
    st.write(f"**Equation:** Price = ${model.coef_[0]:.2f} × Size + ${model.intercept_:.2f}")

if __name__ == "__main__":
    run()