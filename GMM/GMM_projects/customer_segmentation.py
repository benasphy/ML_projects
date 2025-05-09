import streamlit as st
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

def calculate_cluster_metrics(df, cluster_col):
    """Calculate various metrics for each cluster."""
    metrics = {}
    for cluster in df[cluster_col].unique():
        cluster_data = df[df[cluster_col] == cluster]
        metrics[cluster] = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100,
            'avg_age': cluster_data['Age'].mean(),
            'avg_income': cluster_data['Annual_Income'].mean(),
            'avg_spending': cluster_data['Spending_Score'].mean(),
            'avg_frequency': cluster_data['Purchase_Frequency'].mean()
        }
    return metrics

def create_radar_chart(metrics, cluster):
    """Create a radar chart for cluster characteristics."""
    categories = ['Age', 'Income', 'Spending', 'Frequency']
    values = [
        metrics[cluster]['avg_age'] / 100,  # Normalize age
        metrics[cluster]['avg_income'] / 100000,  # Normalize income
        metrics[cluster]['avg_spending'] / 100,  # Normalize spending
        metrics[cluster]['avg_frequency'] / 10  # Normalize frequency
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=f'Cluster {cluster}'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f'Cluster {cluster} Characteristics'
    )
    return fig

def run():
    st.header("Customer Segmentation using GMM")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/GMM)", unsafe_allow_html=True)

    # Load dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using sample customer data")
        # Generate sample customer data
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'Age': np.random.normal(35, 10, n_samples),
            'Annual_Income': np.random.normal(50000, 15000, n_samples),
            'Spending_Score': np.random.normal(50, 20, n_samples),
            'Purchase_Frequency': np.random.normal(5, 2, n_samples)
        })

    # Data Overview
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Number of customers: {len(df)}")
        st.write("Features:", ", ".join(df.columns))
        
        # Basic statistics
        st.write("\n**Basic Statistics:**")
        st.write(df.describe().round(2))
    
    with col2:
        # Feature distributions
        st.write("**Feature Distributions:**")
        fig = px.box(df, title='Feature Distributions')
        st.plotly_chart(fig)

    # Data preprocessing
    X = df.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GMM parameters
    st.subheader("GMM Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_components = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
    with col2:
        covariance_type = st.selectbox("Covariance Type", ['full', 'tied', 'diag', 'spherical'])

    # Train GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
    clusters = gmm.fit_predict(X_scaled)

    # Add clusters to dataframe
    df['Cluster'] = clusters

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    # Cluster Analysis
    st.subheader("Cluster Analysis")
    
    # Calculate cluster metrics
    cluster_metrics = calculate_cluster_metrics(df, 'Cluster')
    
    # Cluster distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names='Cluster', title='Cluster Distribution')
        st.plotly_chart(fig)
    
    with col2:
        fig = px.bar(x=list(cluster_metrics.keys()),
                    y=[m['size'] for m in cluster_metrics.values()],
                    title='Number of Customers per Cluster')
        st.plotly_chart(fig)

    # Cluster Visualizations
    st.subheader("Cluster Visualizations")
    
    # 2D Scatter plot
    fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                    title='Customer Segments (PCA Visualization)',
                    labels={'PCA1': 'First Principal Component',
                           'PCA2': 'Second Principal Component'})
    st.plotly_chart(fig)

    # 3D Scatter plot
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=df['Age'],
        y=df['Annual_Income'],
        z=df['Spending_Score'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['Cluster'],
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    fig_3d.update_layout(
        title='3D Customer Segmentation',
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Annual Income',
            zaxis_title='Spending Score'
        )
    )
    st.plotly_chart(fig_3d)

    # Cluster Characteristics
    st.subheader("Cluster Characteristics")
    
    # Feature distributions by cluster
    for feature in ['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency']:
        fig = px.box(df, x='Cluster', y=feature,
                    title=f'{feature} Distribution by Cluster')
        st.plotly_chart(fig)

    # Radar charts for each cluster
    st.write("**Cluster Profiles:**")
    for cluster in range(n_components):
        fig = create_radar_chart(cluster_metrics, cluster)
        st.plotly_chart(fig)

    # Model Evaluation
    st.subheader("Model Evaluation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BIC Score", f"{gmm.bic(X_scaled):.2f}")
    with col2:
        st.metric("AIC Score", f"{gmm.aic(X_scaled):.2f}")
    with col3:
        st.metric("Convergence Iterations", gmm.n_iter_)

    # Detailed Cluster Interpretation
    st.subheader("Detailed Cluster Interpretation")
    for cluster in range(n_components):
        with st.expander(f"Cluster {cluster} Details"):
            cluster_data = df[df['Cluster'] == cluster]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Size and Distribution:**")
                st.write(f"Number of customers: {len(cluster_data)}")
                st.write(f"Percentage of total: {len(cluster_data)/len(df)*100:.1f}%")
            
            with col2:
                st.write("**Average Characteristics:**")
                st.write(f"Age: {cluster_data['Age'].mean():.1f} years")
                st.write(f"Annual Income: ${cluster_data['Annual_Income'].mean():,.0f}")
                st.write(f"Spending Score: {cluster_data['Spending_Score'].mean():.1f}")
                st.write(f"Purchase Frequency: {cluster_data['Purchase_Frequency'].mean():.1f}")
            
            # Feature correlations
            st.write("**Feature Correlations:**")
            corr = cluster_data[['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency']].corr()
            fig = px.imshow(corr, title=f'Feature Correlations - Cluster {cluster}')
            st.plotly_chart(fig)

if __name__ == "__main__":
    run() 