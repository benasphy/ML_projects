import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.header("Customer Segmentation using K-Means Clustering")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/K-Means)", unsafe_allow_html=True)

    # Load or generate dataset
    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using sample customer data")
        # Generate sample customer data
        np.random.seed(42)
        n_customers = 1000
        
        data = {
            'Customer_ID': range(1, n_customers + 1),
            'Age': np.random.randint(18, 70, n_customers),
            'Annual_Income': np.random.randint(20000, 150000, n_customers),
            'Spending_Score': np.random.randint(1, 100, n_customers),
            'Purchase_Frequency': np.random.randint(1, 50, n_customers),
            'Average_Order_Value': np.random.randint(50, 500, n_customers),
            'Days_Since_Last_Purchase': np.random.randint(0, 365, n_customers)
        }
        df = pd.DataFrame(data)

    # Display data info
    st.subheader("Dataset Information")
    st.write(f"Number of customers: {len(df)}")
    st.write("Sample data:")
    st.dataframe(df.head())

    # Feature selection
    st.subheader("Feature Selection")
    features = ['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency', 
                'Average_Order_Value', 'Days_Since_Last_Purchase']
    selected_features = st.multiselect("Select features for clustering", features, 
                                     default=['Annual_Income', 'Spending_Score'])

    if len(selected_features) >= 2:
        # Prepare data
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means parameters
        st.subheader("Clustering Parameters")
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=6, value=4)
        random_state = st.slider("Random State", min_value=0, max_value=100, value=42)

        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        df['Segment'] = kmeans.fit_predict(X_scaled)

        # Label segments based on average income and spending score
        cluster_means = df.groupby('Segment')[['Annual_Income', 'Spending_Score']].mean()
        segment_labels = {
            0: 'High Value',
            1: 'Low Value',
            2: 'High Potential',
            3: 'At Risk'
        }
        df['Segment'] = df['Segment'].map(segment_labels)

        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['PCA1'] = X_pca[:, 0]
        df['PCA2'] = X_pca[:, 1]

        # PCA Scatter plot
        fig = px.scatter(df, x='PCA1', y='PCA2', color='Segment',
                        hover_data=selected_features,
                        title='Customer Segments (PCA Visualization)')
        st.plotly_chart(fig)

        # Cluster Analysis
        st.subheader("Segment Analysis")
        for segment in df['Segment'].unique():
            st.write(f"\n{segment} Customers:")
            segment_data = df[df['Segment'] == segment]
            st.write(f"Number of customers: {len(segment_data)}")
            
            # Display segment statistics
            stats = segment_data[selected_features].describe()
            st.write("Segment Statistics:")
            st.dataframe(stats)

        # Feature importance visualization
        st.subheader("Feature Importance by Segment")
        segment_means = df.groupby('Segment')[selected_features].mean()
        fig = px.bar(segment_means, title='Average Feature Values by Segment',
                    labels={'value': 'Average Value', 'variable': 'Feature'})
        st.plotly_chart(fig)

        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        correlation_matrix = df[selected_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(fig)

        # Segment distribution
        st.subheader("Segment Distribution")
        segment_counts = df['Segment'].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title='Customer Segment Distribution')
        st.plotly_chart(fig)

        # Interactive prediction
        st.subheader("Predict Segment for New Customer")
        input_values = {}
        for feature in selected_features:
            if feature == 'Age':
                input_values[feature] = st.number_input(feature, min_value=18, max_value=70, value=35)
            elif feature == 'Annual_Income':
                input_values[feature] = st.number_input(feature, min_value=20000, max_value=150000, value=50000)
            elif feature == 'Spending_Score':
                input_values[feature] = st.number_input(feature, min_value=1, max_value=100, value=50)
            elif feature == 'Purchase_Frequency':
                input_values[feature] = st.number_input(feature, min_value=1, max_value=50, value=10)
            elif feature == 'Average_Order_Value':
                input_values[feature] = st.number_input(feature, min_value=50, max_value=500, value=100)
            elif feature == 'Days_Since_Last_Purchase':
                input_values[feature] = st.number_input(feature, min_value=0, max_value=365, value=30)

        if st.button("Predict Customer Segment"):
            # Create input array with only the selected features
            new_customer = np.array([[input_values[feature] for feature in selected_features]])
            new_customer_scaled = scaler.transform(new_customer)
            prediction = kmeans.predict(new_customer_scaled)[0]
            segment = segment_labels[prediction]
            st.success(f"Predicted Customer Segment: {segment}")

if __name__ == "__main__":
    run() 