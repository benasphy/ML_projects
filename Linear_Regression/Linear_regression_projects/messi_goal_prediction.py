import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import importlib

def calculate_residuals(y_true, y_pred):
    """Calculate and return residuals."""
    return y_true - y_pred

def run():
    st.header("Messi Goal Prediction")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Linear_Regression)", unsafe_allow_html=True)

    # Example data (Messi's goals per season)
    seasons = np.array([2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    goals = np.array([1, 8, 17, 16, 38, 47, 53, 73, 91, 45, 58, 48, 59, 54, 51, 50, 38, 43, 35, 28])
    matches = np.array([9, 17, 26, 28, 51, 53, 55, 60, 60, 46, 57, 49, 52, 54, 50, 50, 47, 41, 41, 41])

    # Create DataFrame
    df = pd.DataFrame({
        'Season': seasons,
        'Goals': goals,
        'Matches': matches
    })

    # Data Overview
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Number of seasons: {len(df)}")
        st.write("\n**Basic Statistics:**")
        st.write(df.describe().round(2))
    
    with col2:
        # Distribution plots
        fig = px.box(df, y=['Goals', 'Matches'], title='Goals and Matches Distribution')
        st.plotly_chart(fig)

    # Data Analysis
    st.subheader("Data Analysis")
    
    # Correlation analysis
    correlation = df['Season'].corr(df['Goals'])
    st.write(f"**Correlation between Season and Goals:** {correlation:.3f}")
    
    # Scatter plot with trend line
    fig = px.scatter(df, x='Season', y='Goals',
                    title='Messi Goals per Season',
                    labels={'Season': 'Season',
                           'Goals': 'Number of Goals'})
    fig.add_trace(go.Scatter(x=df['Season'],
                            y=stats.linregress(df['Season'], df['Goals'])[0] * df['Season'] + 
                              stats.linregress(df['Season'], df['Goals'])[1],
                            mode='lines',
                            name='Trend Line'))
    st.plotly_chart(fig)

    # Reshape data
    X = seasons.reshape(-1, 1)
    y = goals

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
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_pred)):.2f} goals")
    with col3:
        st.metric("MAE", f"{mean_absolute_error(y, y_pred):.2f} goals")

    # Residual Analysis
    st.subheader("Residual Analysis")
    residuals = calculate_residuals(y, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        # Residuals vs Predicted
        fig = px.scatter(x=y_pred, y=residuals,
                        title='Residuals vs Predicted Values',
                        labels={'x': 'Predicted Goals',
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
    st.subheader("Goal Prediction")
    
    # Create tabs for different prediction types
    tab1, tab2 = st.tabs(["Season Prediction", "Match-based Prediction"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            season = st.number_input("Enter season year:",
                                   min_value=2009,
                                   max_value=2030,
                                   value=2024,
                                   step=1)
            
            # Calculate prediction
            prediction = model.predict([[season]])[0]
            confidence_interval = 1.96 * np.sqrt(mean_squared_error(y, y_pred))
            
            st.metric("Predicted Goals",
                     f"{prediction:.1f}",
                     f"±{confidence_interval:.1f}")
        
        with col2:
            # Prediction visualization
            fig = go.Figure()
            
            # Add actual data points
            fig.add_trace(go.Scatter(
                x=df['Season'],
                y=df['Goals'],
                mode='markers',
                name='Actual Goals',
                marker=dict(color='blue')
            ))
            
            # Add regression line
            x_range = np.linspace(min(df['Season']), max(df['Season']), 100)
            y_range = model.predict(x_range.reshape(-1, 1))
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                name='Trend Line',
                line=dict(color='red')
            ))
            
            # Add prediction point
            fig.add_trace(go.Scatter(
                x=[season],
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
                title='Messi Goals Prediction',
                xaxis_title='Season',
                yaxis_title='Number of Goals',
                showlegend=True
            )
            st.plotly_chart(fig)
    
    with tab2:
        # Calculate goals per match ratio
        df['Goals_per_Match'] = df['Goals'] / df['Matches']
        avg_goals_per_match = df['Goals_per_Match'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            matches = st.number_input("Enter number of matches:",
                                    min_value=1,
                                    max_value=60,
                                    value=30,
                                    step=1)
            
            # Calculate prediction
            predicted_goals = matches * avg_goals_per_match
            confidence_interval = 1.96 * np.sqrt(mean_squared_error(df['Goals'], df['Matches'] * avg_goals_per_match))
            
            st.metric("Predicted Goals",
                     f"{predicted_goals:.1f}",
                     f"±{confidence_interval:.1f}")
            
            st.write(f"**Goals per match ratio:** {avg_goals_per_match:.2f}")
        
        with col2:
            # Goals per match distribution
            fig = px.box(df, y='Goals_per_Match',
                        title='Goals per Match Distribution',
                        labels={'Goals_per_Match': 'Goals per Match'})
            st.plotly_chart(fig)

    # Model Information
    st.subheader("Model Information")
    st.write(f"**Slope (Goals per season):** {model.coef_[0]:.2f}")
    st.write(f"**Intercept:** {model.intercept_:.2f}")
    st.write(f"**Equation:** Goals = {model.coef_[0]:.2f} × Season + {model.intercept_:.2f}")

    # Career Analysis
    st.subheader("Career Analysis")
    
    # Calculate career statistics
    total_goals = df['Goals'].sum()
    avg_goals = df['Goals'].mean()
    max_goals = df['Goals'].max()
    max_goals_season = df.loc[df['Goals'].idxmax(), 'Season']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Goals", f"{total_goals}")
    with col2:
        st.metric("Average Goals/Season", f"{avg_goals:.1f}")
    with col3:
        st.metric("Most Goals in a Season", f"{max_goals}")
    with col4:
        st.metric("Best Season", f"{max_goals_season}")

    # Goals per season bar chart
    fig = px.bar(df, x='Season', y='Goals',
                 title='Messi Goals per Season',
                 labels={'Season': 'Season',
                        'Goals': 'Number of Goals'})
    st.plotly_chart(fig)

    # Career Phases Analysis
    st.subheader("Career Phases Analysis")
    
    # Define career phases
    phases = {
        'Early Career (2004-2008)': df[df['Season'] <= 2008],
        'Peak Years (2009-2015)': df[(df['Season'] > 2008) & (df['Season'] <= 2015)],
        'Later Career (2016-2023)': df[df['Season'] > 2015]
    }
    
    # Calculate phase statistics
    phase_stats = pd.DataFrame({
        'Phase': list(phases.keys()),
        'Average Goals': [phase['Goals'].mean() for phase in phases.values()],
        'Total Goals': [phase['Goals'].sum() for phase in phases.values()],
        'Best Season': [phase.loc[phase['Goals'].idxmax(), 'Season'] for phase in phases.values()],
        'Most Goals': [phase['Goals'].max() for phase in phases.values()]
    })
    
    st.write(phase_stats)
    
    # Plot phase comparison
    fig = px.bar(phase_stats, x='Phase', y='Average Goals',
                 title='Average Goals per Season by Career Phase',
                 labels={'Phase': 'Career Phase',
                        'Average Goals': 'Average Goals per Season'})
    st.plotly_chart(fig)

    # Career Milestones
    st.subheader("Career Milestones")
    
    # Calculate cumulative goals
    df['Cumulative_Goals'] = df['Goals'].cumsum()
    
    # Plot cumulative goals
    fig = px.line(df, x='Season', y='Cumulative_Goals',
                  title='Cumulative Goals Over Career',
                  labels={'Season': 'Season',
                         'Cumulative_Goals': 'Total Goals'})
    st.plotly_chart(fig)
    
    # Highlight key milestones
    milestones = [100, 200, 300, 400, 500, 600, 700]
    for milestone in milestones:
        if milestone <= df['Cumulative_Goals'].max():
            season = df[df['Cumulative_Goals'] >= milestone]['Season'].iloc[0]
            st.write(f"- Reached {milestone} goals in {season}")

if __name__ == "__main__":
    if st.button("Go to App"):
        selected_path = algorithm_paths[selected_algorithm]
        module_path = selected_path.replace("/", ".").replace(".py", "")
        module = importlib.import_module(module_path)
        module.run()