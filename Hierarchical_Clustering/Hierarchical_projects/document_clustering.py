import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.header("Document Clustering using Hierarchical Clustering")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/Hierarchical_Clustering)", unsafe_allow_html=True)

    # Load dataset
    uploaded_file = st.file_uploader("Upload a CSV file with text documents", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Assuming the text column is named 'text'
        texts = df['text'].values
    else:
        st.info("Using sample document data")
        # Generate sample documents
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing helps computers understand text",
            "Computer vision enables machines to interpret images",
            "Data science combines statistics and programming",
            "Big data refers to large and complex datasets",
            "Cloud computing provides on-demand computing resources",
            "Internet of Things connects physical devices to the internet",
            "Cybersecurity protects systems from digital attacks",
            "Blockchain is a distributed ledger technology",
            "Quantum computing uses quantum bits for calculations",
            "Augmented reality overlays digital content on the real world",
            "Virtual reality creates immersive digital environments",
            "Robotics combines mechanical and electronic systems",
            "5G technology enables faster wireless communication"
        ]

    # Text preprocessing and vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    # Hierarchical Clustering parameters
    st.subheader("Clustering Parameters")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
    linkage_method = st.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single'])

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )
    clusters = clustering.fit_predict(X.toarray())

    # Create dendrogram
    st.subheader("Dendrogram")
    Z = linkage(X.toarray(), method=linkage_method)
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    st.pyplot(fig)

    # Document similarity matrix
    st.subheader("Document Similarity Matrix")
    similarity_matrix = cosine_similarity(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title('Document Similarity Matrix')
    st.pyplot(fig)

    # Cluster visualization using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    
    # Create DataFrame for visualization
    viz_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters,
        'Document': [f"Doc {i+1}" for i in range(len(texts))]
    })

    # 2D Scatter plot
    fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster',
                    hover_data=['Document'],
                    title='Document Clusters (PCA Visualization)')
    st.plotly_chart(fig)

    # Cluster analysis
    st.subheader("Cluster Analysis")
    for cluster in range(n_clusters):
        st.write(f"\nCluster {cluster}:")
        cluster_docs = [texts[i] for i in range(len(texts)) if clusters[i] == cluster]
        st.write(f"Number of documents: {len(cluster_docs)}")
        st.write("Documents in this cluster:")
        for doc in cluster_docs:
            st.write(f"- {doc}")

    # Cluster statistics
    st.subheader("Cluster Statistics")
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    fig = px.bar(x=cluster_sizes.index, y=cluster_sizes.values,
                 title='Number of Documents per Cluster',
                 labels={'x': 'Cluster', 'y': 'Number of Documents'})
    st.plotly_chart(fig)

    # Word clouds for each cluster
    st.subheader("Word Clouds by Cluster")
    from wordcloud import WordCloud
    
    for cluster in range(n_clusters):
        cluster_docs = [texts[i] for i in range(len(texts)) if clusters[i] == cluster]
        if cluster_docs:
            text = ' '.join(cluster_docs)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.title(f'Word Cloud - Cluster {cluster}')
            st.pyplot(fig)

if __name__ == "__main__":
    run() 