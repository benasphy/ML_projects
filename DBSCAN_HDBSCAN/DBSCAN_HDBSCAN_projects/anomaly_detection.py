import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.header("Anomaly Detection using DBSCAN")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/DBSCAN_HDBSCAN)", unsafe_allow_html=True)

    # Load or generate dataset
    uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using sample transaction data")
        # Generate sample transaction data
        np.random.seed(42)
        n_transactions = 1000
        
        # Generate normal transactions
        normal_amounts = np.random.normal(100, 20, int(n_transactions * 0.95))
        normal_times = np.random.normal(12, 2, int(n_transactions * 0.95))
        
        # Generate anomalous transactions
        anomaly_amounts = np.random.uniform(500, 1000, int(n_transactions * 0.05))
        anomaly_times = np.random.uniform(0, 24, int(n_transactions * 0.05))
        
        # Combine normal and anomalous data
        amounts = np.concatenate([normal_amounts, anomaly_amounts])
        times = np.concatenate([normal_times, anomaly_times])
        
        data = {
            'Transaction_ID': range(1, n_transactions + 1),
            'Amount': amounts,
            'Time': times,
            'Location_X': np.random.normal(0, 1, n_transactions),
            'Location_Y': np.random.normal(0, 1, n_transactions),
            'Merchant_Category': np.random.choice(['Retail', 'Food', 'Travel', 'Other'], n_transactions)
        }
        df = pd.DataFrame(data)

    # Display data info
    st.subheader("Dataset Information")
    st.write(f"Number of transactions: {len(df)}")
    st.write("Sample data:")
    st.dataframe(df.head())

    # Feature selection
    st.subheader("Feature Selection")
    features = ['Amount', 'Time', 'Location_X', 'Location_Y']
    selected_features = st.multiselect("Select features for anomaly detection", features, 
                                     default=['Amount', 'Time'])

    if len(selected_features) >= 2:
        # Prepare data
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # DBSCAN parameters
        st.subheader("DBSCAN Parameters")
        eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples = st.slider("Minimum Samples", min_value=2, max_value=20, value=5)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['Cluster'] = dbscan.fit_predict(X_scaled)
        
        # Label clusters
        df['Anomaly'] = df['Cluster'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['PCA1'] = X_pca[:, 0]
        df['PCA2'] = X_pca[:, 1]

        # PCA Scatter plot
        fig = px.scatter(df, x='PCA1', y='PCA2', color='Anomaly',
                        hover_data=selected_features,
                        title='Transaction Clusters (PCA Visualization)')
        st.plotly_chart(fig)

        # Anomaly Analysis
        st.subheader("Anomaly Analysis")
        anomaly_count = len(df[df['Anomaly'] == 'Anomaly'])
        st.write(f"Number of anomalies detected: {anomaly_count}")
        st.write(f"Percentage of anomalies: {(anomaly_count/len(df))*100:.2f}%")

        # Display anomaly statistics
        st.write("\nAnomaly Statistics:")
        anomaly_stats = df[df['Anomaly'] == 'Anomaly'][selected_features].describe()
        st.dataframe(anomaly_stats)

        # Feature importance visualization
        st.subheader("Feature Distribution by Cluster")
        for feature in selected_features:
            fig = px.box(df, x='Anomaly', y=feature, title=f'{feature} Distribution by Cluster')
            st.plotly_chart(fig)

        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        correlation_matrix = df[selected_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(fig)

        # Interactive prediction
        st.subheader("Check New Transaction")
        input_values = {}
        for feature in selected_features:
            if feature == 'Amount':
                input_values[feature] = st.number_input(feature, min_value=0.0, max_value=1000.0, value=100.0)
            elif feature == 'Time':
                input_values[feature] = st.number_input(feature, min_value=0.0, max_value=24.0, value=12.0)
            elif feature == 'Location_X':
                input_values[feature] = st.number_input(feature, min_value=-3.0, max_value=3.0, value=0.0)
            elif feature == 'Location_Y':
                input_values[feature] = st.number_input(feature, min_value=-3.0, max_value=3.0, value=0.0)

        if st.button("Check for Anomaly"):
            # Create input array with only the selected features
            new_transaction = np.array([[input_values[feature] for feature in selected_features]])
            new_transaction_scaled = scaler.transform(new_transaction)
            prediction = dbscan.fit_predict(np.vstack([X_scaled, new_transaction_scaled]))[-1]
            result = "Anomaly" if prediction == -1 else "Normal"
            st.success(f"Transaction is classified as: {result}")

if __name__ == "__main__":
    run() 