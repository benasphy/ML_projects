import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import time

def normal_equation(X, y):
    """Calculate parameters using normal equation."""
    start_time = time.time()
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    end_time = time.time()
    return theta, end_time - start_time

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """Calculate parameters using gradient descent."""
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []
    
    start_time = time.time()
    for i in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * X.T @ errors
        theta = theta - learning_rate * gradient
        
        # Calculate cost
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
        
        # Check for convergence
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < 1e-10:
            break
    
    end_time = time.time()
    return theta, cost_history, end_time - start_time, i + 1

def run():
    st.header("Normal Equation vs Gradient Descent Comparison")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Linear_Regression)", unsafe_allow_html=True)

    # Data Generation
    st.subheader("Data Generation")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.slider("Number of samples", min_value=10, max_value=1000, value=100, step=10)
    with col2:
        noise_level = st.slider("Noise level", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    with col3:
        true_slope = st.slider("True slope", min_value=0.1, max_value=5.0, value=2.0, step=0.1)

    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10
    y = true_slope * X.ravel() + 1 + np.random.randn(n_samples) * noise_level

    # Add bias term
    X_b = np.c_[np.ones((n_samples, 1)), X]

    # Data Overview
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Number of samples: {n_samples}")
        st.write("\n**Basic Statistics:**")
        df = pd.DataFrame({'X': X.ravel(), 'y': y})
        st.write(df.describe().round(2))
    
    with col2:
        # Distribution plots
        fig = px.box(df, y=['X', 'y'],
                    title='Feature and Target Distributions')
        st.plotly_chart(fig)

    # Data Visualization
    st.subheader("Data Visualization")
    fig = px.scatter(x=X.ravel(), y=y,
                    title='Sample Data',
                    labels={'x': 'Feature', 'y': 'Target'})
    st.plotly_chart(fig)

    # Parameter Selection
    st.subheader("Gradient Descent Parameters")
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    with col2:
        iterations = st.slider("Number of Iterations", min_value=100, max_value=5000, value=1000, step=100)

    # Calculate parameters using both methods
    theta_normal, time_normal = normal_equation(X_b, y)
    theta_gd, cost_history, time_gd, actual_iterations = gradient_descent(X_b, y, learning_rate, iterations)

    # Results Comparison
    st.subheader("Results Comparison")
    
    # Performance Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Normal Equation Time", f"{time_normal:.6f} seconds")
    with col2:
        st.metric("Gradient Descent Time", f"{time_gd:.6f} seconds")
    with col3:
        st.metric("Time Difference", f"{abs(time_normal - time_gd):.6f} seconds")

    # Parameter Comparison
    st.write("**Parameter Comparison:**")
    params_df = pd.DataFrame({
        'Method': ['Normal Equation', 'Gradient Descent', 'True Values'],
        'Intercept': [theta_normal[0], theta_gd[0], 1.0],
        'Slope': [theta_normal[1], theta_gd[1], true_slope],
        'Execution Time (s)': [time_normal, time_gd, 'N/A'],
        'Iterations': ['N/A', actual_iterations, 'N/A']
    })
    st.write(params_df)

    # Cost History Plot
    st.subheader("Gradient Descent Cost History")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=cost_history,
        mode='lines',
        name='Cost'
    ))
    fig.update_layout(
        title='Cost vs Iterations',
        xaxis_title='Iteration',
        yaxis_title='Cost',
        showlegend=True
    )
    st.plotly_chart(fig)

    # Convergence Analysis
    st.subheader("Convergence Analysis")
    st.write("**Cost History Analysis:**")
    st.write(f"- Initial Cost: {cost_history[0]:.4f}")
    st.write(f"- Final Cost: {cost_history[-1]:.4f}")
    st.write(f"- Cost Reduction: {((cost_history[0] - cost_history[-1]) / cost_history[0] * 100):.2f}%")
    st.write(f"- Iterations to Convergence: {actual_iterations}")
    
    # Plot cost reduction
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=[cost_history[0], cost_history[-1]],
        mode='lines+markers',
        name='Cost Reduction'
    ))
    fig.update_layout(
        title='Cost Reduction',
        xaxis_title='Iteration',
        yaxis_title='Cost',
        showlegend=True
    )
    st.plotly_chart(fig)

    # Final Models Visualization
    st.subheader("Model Comparison")
    
    # Generate points for plotting
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_b = np.c_[np.ones((100, 1)), X_plot]
    
    # Calculate predictions
    y_normal = X_plot_b @ theta_normal
    y_gd = X_plot_b @ theta_gd
    y_true = true_slope * X_plot.ravel() + 1
    
    # Plot
    fig = go.Figure()
    
    # Add data points
    fig.add_trace(go.Scatter(
        x=X.ravel(),
        y=y,
        mode='markers',
        name='Data Points',
        marker=dict(color='blue')
    ))
    
    # Add true line
    fig.add_trace(go.Scatter(
        x=X_plot.ravel(),
        y=y_true,
        mode='lines',
        name='True Line',
        line=dict(color='black', dash='dash')
    ))
    
    # Add normal equation line
    fig.add_trace(go.Scatter(
        x=X_plot.ravel(),
        y=y_normal,
        mode='lines',
        name='Normal Equation',
        line=dict(color='red')
    ))
    
    # Add gradient descent line
    fig.add_trace(go.Scatter(
        x=X_plot.ravel(),
        y=y_gd,
        mode='lines',
        name='Gradient Descent',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Feature',
        yaxis_title='Target',
        showlegend=True
    )
    st.plotly_chart(fig)

    # Model Evaluation
    st.subheader("Model Evaluation")
    
    # Calculate predictions for both methods
    y_pred_normal = X_b @ theta_normal
    y_pred_gd = X_b @ theta_gd
    
    # Calculate metrics
    metrics_df = pd.DataFrame({
        'Method': ['Normal Equation', 'Gradient Descent'],
        'R² Score': [r2_score(y, y_pred_normal), r2_score(y, y_pred_gd)],
        'RMSE': [np.sqrt(mean_squared_error(y, y_pred_normal)),
                np.sqrt(mean_squared_error(y, y_pred_gd))],
        'Parameters': [f"θ₀={theta_normal[0]:.4f}, θ₁={theta_normal[1]:.4f}",
                      f"θ₀={theta_gd[0]:.4f}, θ₁={theta_gd[1]:.4f}"]
    })
    st.write(metrics_df)

    # Method Comparison
    st.subheader("Method Comparison")
    
    # Create tabs for different aspects
    tab1, tab2, tab3 = st.tabs(["Advantages", "Disadvantages", "Use Cases"])
    
    with tab1:
        st.write("**Normal Equation:**")
        st.write("- Direct solution, no iterations needed")
        st.write("- Guaranteed to find the optimal solution")
        st.write("- Works well for small to medium datasets")
        st.write("- No hyperparameters to tune")
        
        st.write("\n**Gradient Descent:**")
        st.write("- Works well with large datasets")
        st.write("- Can be implemented online (streaming data)")
        st.write("- Memory efficient")
        st.write("- Can handle non-linear models")
    
    with tab2:
        st.write("**Normal Equation:**")
        st.write("- Computationally expensive for large datasets")
        st.write("- Requires matrix inversion")
        st.write("- Memory intensive")
        st.write("- Can be numerically unstable")
        
        st.write("\n**Gradient Descent:**")
        st.write("- Requires tuning of learning rate")
        st.write("- May converge to local minimum")
        st.write("- Needs multiple iterations")
        st.write("- Convergence can be slow")
    
    with tab3:
        st.write("**Normal Equation is best for:**")
        st.write("- Small to medium datasets (n < 10,000)")
        st.write("- When exact solution is needed")
        st.write("- When computational resources are available")
        st.write("- When matrix operations are efficient")
        
        st.write("\n**Gradient Descent is best for:**")
        st.write("- Large datasets")
        st.write("- Online learning scenarios")
        st.write("- When memory is limited")
        st.write("- When approximate solution is acceptable")

    # Interactive Prediction
    st.subheader("Interactive Prediction")
    input_value = st.number_input("Enter feature value for prediction:",
                                min_value=float(X.min()),
                                max_value=float(X.max()),
                                value=float(X.mean()),
                                step=0.1)
    
    # Calculate predictions
    input_b = np.array([[1, input_value]])
    pred_normal = input_b @ theta_normal
    pred_gd = input_b @ theta_gd
    pred_true = true_slope * input_value + 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Normal Equation Prediction", f"{pred_normal[0]:.4f}")
    with col2:
        st.metric("Gradient Descent Prediction", f"{pred_gd[0]:.4f}")
    with col3:
        st.metric("True Value", f"{pred_true:.4f}")
    
    # Plot prediction comparison
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[input_value],
        y=[pred_normal],
        mode='markers',
        name='Normal Equation',
        marker=dict(color='red', size=12)
    ))
    fig.add_trace(go.Scatter(
        x=[input_value],
        y=[pred_gd],
        mode='markers',
        name='Gradient Descent',
        marker=dict(color='green', size=12)
    ))
    fig.add_trace(go.Scatter(
        x=[input_value],
        y=[pred_true],
        mode='markers',
        name='True Value',
        marker=dict(color='black', size=12)
    ))
    fig.update_layout(
        title='Prediction Comparison',
        xaxis_title='Feature Value',
        yaxis_title='Predicted Value',
        showlegend=True
    )
    st.plotly_chart(fig)

if __name__ == "__main__":
    run()