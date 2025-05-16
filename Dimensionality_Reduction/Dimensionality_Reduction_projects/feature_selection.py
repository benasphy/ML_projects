import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def generate_sample_data(n_samples=1000):
    # Generate features with different patterns
    np.random.seed(42)
    
    # Generate correlated features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = x1 + np.random.normal(0, 0.5, n_samples)
    x3 = x1 - x2 + np.random.normal(0, 0.3, n_samples)
    
    # Generate independent features
    x4 = np.random.normal(0, 1, n_samples)
    x5 = np.random.normal(0, 1, n_samples)
    
    # Generate target variable
    y = 2*x1 + 3*x2 - x3 + np.random.normal(0, 0.5, n_samples)
    
    # Create DataFrame
    data = {
        'Feature1': x1,
        'Feature2': x2,
        'Feature3': x3,
        'Feature4': x4,
        'Feature5': x5,
        'Target': y
    }
    
    return pd.DataFrame(data)

def run():
    st.header("Feature Selection using Dimensionality Reduction")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/Dimensionality_Reduction)", unsafe_allow_html=True)

    # Load or generate dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using sample dataset")
        df = generate_sample_data()

    # Display data info
    st.subheader("Dataset Information")
    st.write(f"Number of samples: {len(df)}")
    st.write(f"Number of features: {len(df.columns) - 1}")  # Excluding target
    st.write("Sample data:")
    st.dataframe(df.head())

    # Feature selection
    st.subheader("Feature Selection")
    features = [col for col in df.columns if col != 'Target']
    selected_features = st.multiselect("Select features for analysis", features, default=features)

    if len(selected_features) >= 2:
        # Prepare data
        X = df[selected_features]
        y = df['Target'] if 'Target' in df.columns else None
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dimensionality reduction methods
        st.subheader("Dimensionality Reduction Methods")
        method = st.selectbox(
            "Select method",
            ["PCA", "t-SNE"]
        )

        if method == "PCA":
            # PCA parameters
            n_components = st.slider("Number of Components", min_value=1, max_value=len(selected_features), value=2)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create DataFrame with PCA results
            pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
            if y is not None:
                pca_df['Target'] = y
            
            # Plot PCA results
            st.subheader("PCA Results")
            
            # Scatter plot
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target' if y is not None else None,
                           title="PCA Visualization",
                           labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                                 'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'})
            st.plotly_chart(fig)
            
            # Explained variance
            fig = px.bar(x=range(1, len(pca.explained_variance_ratio_) + 1),
                        y=pca.explained_variance_ratio_,
                        title="Explained Variance by Component",
                        labels={'x': 'Component', 'y': 'Explained Variance'})
            st.plotly_chart(fig)
            
            # Cumulative explained variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            fig = px.line(x=range(1, len(cumulative_variance) + 1),
                         y=cumulative_variance,
                         title="Cumulative Explained Variance",
                         labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'})
            st.plotly_chart(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=selected_features
            )
            st.dataframe(feature_importance)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(feature_importance, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)

        else:  # t-SNE
            # t-SNE parameters
            perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30)
            learning_rate = st.slider("Learning Rate", min_value=10, max_value=1000, value=200)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
            X_tsne = tsne.fit_transform(X_scaled)
            
            # Create DataFrame with t-SNE results
            tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE1', 't-SNE2'])
            if y is not None:
                tsne_df['Target'] = y
            
            # Plot t-SNE results
            st.subheader("t-SNE Results")
            fig = px.scatter(tsne_df, x='t-SNE1', y='t-SNE2', color='Target' if y is not None else None,
                           title="t-SNE Visualization")
            st.plotly_chart(fig)
            
            # Feature correlation
            st.subheader("Feature Correlation")
            correlation_matrix = df[selected_features].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    run() 