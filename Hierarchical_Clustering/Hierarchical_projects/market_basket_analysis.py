import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.header("Market Basket Analysis using Hierarchical Clustering")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Hierarchical_Clustering)", unsafe_allow_html=True)

    # Load dataset
    uploaded_file = st.file_uploader("Upload a CSV file with purchase data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using sample market basket data")
        # Generate sample market basket data
        np.random.seed(42)
        n_transactions = 1000
        n_items = 20
        
        # Generate random purchase patterns
        data = np.random.binomial(1, 0.3, (n_transactions, n_items))
        
        # Create item names
        item_names = [f'Item_{i+1}' for i in range(n_items)]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=item_names)

    # Display data info
    st.subheader("Dataset Information")
    st.write(f"Number of transactions: {len(df)}")
    st.write(f"Number of items: {len(df.columns)}")
    st.write("Sample data:")
    st.dataframe(df.head())

    # Hierarchical Clustering parameters
    st.subheader("Clustering Parameters")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
    linkage_method = st.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single'])

    # Prepare data for clustering
    X = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )
    clusters = clustering.fit_predict(X_scaled)

    # Create dendrogram
    st.subheader("Dendrogram")
    Z = linkage(X_scaled, method=linkage_method)
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    st.pyplot(fig)

    # Item correlation matrix
    st.subheader("Item Correlation Matrix")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title('Item Correlation Matrix')
    st.pyplot(fig)

    # Cluster visualization using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame for visualization
    viz_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters
    })

    # 2D Scatter plot
    fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster',
                    title='Transaction Clusters (PCA Visualization)')
    st.plotly_chart(fig)

    # Cluster analysis
    st.subheader("Cluster Analysis")
    for cluster in range(n_clusters):
        st.write(f"\nCluster {cluster}:")
        cluster_data = df[clusters == cluster]
        st.write(f"Number of transactions: {len(cluster_data)}")
        
        # Calculate item frequencies in this cluster
        item_freq = cluster_data.mean().sort_values(ascending=False)
        top_items = item_freq[item_freq > 0.1]  # Show items that appear in more than 10% of transactions
        
        st.write("Top items in this cluster:")
        for item, freq in top_items.items():
            st.write(f"- {item}: {freq:.1%} of transactions")

    # Cluster statistics
    st.subheader("Cluster Statistics")
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    fig = px.bar(x=cluster_sizes.index, y=cluster_sizes.values,
                 title='Number of Transactions per Cluster',
                 labels={'x': 'Cluster', 'y': 'Number of Transactions'})
    st.plotly_chart(fig)

    # Item frequency by cluster
    st.subheader("Item Frequency by Cluster")
    cluster_item_freq = df.groupby(clusters).mean()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cluster_item_freq, cmap='YlOrRd')
    plt.title('Item Frequency by Cluster')
    st.pyplot(fig)

    # Association rules
    st.subheader("Association Rules")
    from mlxtend.frequent_patterns import apriori, association_rules
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
    
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    if not rules.empty:
        st.write("Top Association Rules:")
        # Convert frozensets to strings for display
        display_rules = rules.copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        st.dataframe(display_rules.head())
        
        # Create a copy of rules for visualization with string versions of frozensets
        viz_rules = rules.copy()
        viz_rules['antecedents_str'] = viz_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        viz_rules['consequents_str'] = viz_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Visualize rules
        fig = px.scatter(viz_rules, x='support', y='confidence',
                        size='lift', color='lift',
                        hover_data=['antecedents_str', 'consequents_str'],
                        title='Association Rules Visualization')
        st.plotly_chart(fig)

if __name__ == "__main__":
    run() 