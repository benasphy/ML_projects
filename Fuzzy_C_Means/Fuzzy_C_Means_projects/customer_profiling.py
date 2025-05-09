import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def fuzzy_c_means(data, n_clusters, m=2, max_iter=100, error=1e-5):
    # Initialize membership matrix randomly
    n_samples = data.shape[0]
    membership = np.random.random((n_samples, n_clusters))
    membership = membership / membership.sum(axis=1)[:, np.newaxis]
    
    # Initialize centers
    centers = np.zeros((n_clusters, data.shape[1]))
    
    for iteration in range(max_iter):
        # Calculate centers
        for j in range(n_clusters):
            centers[j] = np.sum(membership[:, j:j+1]**m * data, axis=0) / np.sum(membership[:, j:j+1]**m)
        
        # Calculate new membership
        distances = cdist(data, centers)
        new_membership = 1 / (distances ** (2/(m-1)))
        new_membership = new_membership / new_membership.sum(axis=1)[:, np.newaxis]
        
        # Check convergence
        if np.max(np.abs(new_membership - membership)) < error:
            break
            
        membership = new_membership
    
    return centers, membership

def run():
    st.header("Customer Profiling using Fuzzy C-Means")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Fuzzy_C_Means)", unsafe_allow_html=True)

    # Load or generate dataset
    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using sample customer data")
        # Generate sample customer data
        np.random.seed(42)
        n_customers = 1000
        
        # Generate different customer segments
        segments = {
            'High_Value': {'size': 0.2, 'income': (80000, 150000), 'frequency': (20, 40), 'recency': (0, 30)},
            'Regular': {'size': 0.4, 'income': (40000, 80000), 'frequency': (10, 20), 'recency': (30, 90)},
            'Occasional': {'size': 0.3, 'income': (20000, 40000), 'frequency': (5, 10), 'recency': (90, 180)},
            'Inactive': {'size': 0.1, 'income': (0, 20000), 'frequency': (0, 5), 'recency': (180, 365)}
        }
        
        data = {
            'Customer_ID': range(1, n_customers + 1),
            'Annual_Income': [],
            'Purchase_Frequency': [],
            'Days_Since_Last_Purchase': [],
            'Average_Order_Value': [],
            'Website_Time_Spent': [],
            'App_Usage_Frequency': []
        }
        
        for segment, params in segments.items():
            n_segment = int(n_customers * params['size'])
            data['Annual_Income'].extend(np.random.uniform(*params['income'], n_segment))
            data['Purchase_Frequency'].extend(np.random.uniform(*params['frequency'], n_segment))
            data['Days_Since_Last_Purchase'].extend(np.random.uniform(*params['recency'], n_segment))
            data['Average_Order_Value'].extend(np.random.uniform(50, 500, n_segment))
            data['Website_Time_Spent'].extend(np.random.uniform(5, 60, n_segment))
            data['App_Usage_Frequency'].extend(np.random.uniform(1, 30, n_segment))
        
        df = pd.DataFrame(data)

    # Display data info
    st.subheader("Dataset Information")
    st.write(f"Number of customers: {len(df)}")
    st.write("Sample data:")
    st.dataframe(df.head())

    # Feature selection
    st.subheader("Feature Selection")
    features = ['Annual_Income', 'Purchase_Frequency', 'Days_Since_Last_Purchase',
                'Average_Order_Value', 'Website_Time_Spent', 'App_Usage_Frequency']
    selected_features = st.multiselect("Select features for profiling", features, 
                                     default=['Annual_Income', 'Purchase_Frequency', 'Days_Since_Last_Purchase'])

    if len(selected_features) >= 2:
        # Prepare data
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fuzzy C-Means parameters
        st.subheader("Fuzzy C-Means Parameters")
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=6, value=4)
        fuzziness = st.slider("Fuzziness Parameter (m)", min_value=1.1, max_value=3.0, value=2.0, step=0.1)

        # Apply Fuzzy C-Means
        centers, membership = fuzzy_c_means(X_scaled, n_clusters, m=fuzziness)
        
        # Get cluster assignments
        df['Cluster'] = np.argmax(membership, axis=1)
        
        # Label clusters based on average income
        cluster_means = df.groupby('Cluster')['Annual_Income'].mean()
        cluster_order = cluster_means.rank(ascending=False).astype(int) - 1
        cluster_labels = {i: f"Segment {j+1}" for i, j in cluster_order.items()}
        df['Segment'] = df['Cluster'].map(cluster_labels)

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

        # Segment Analysis
        st.subheader("Segment Analysis")
        segment_counts = df['Segment'].value_counts()
        st.write("Segment Distribution:")
        st.write(segment_counts)
        
        # Display segment statistics
        for segment in df['Segment'].unique():
            st.write(f"\n{segment} Customers:")
            segment_data = df[df['Segment'] == segment]
            st.write(f"Number of customers: {len(segment_data)}")
            stats = segment_data[selected_features].describe()
            st.write("Segment Statistics:")
            st.dataframe(stats)

        # Feature importance visualization
        st.subheader("Feature Distribution by Segment")
        for feature in selected_features:
            fig = px.box(df, x='Segment', y=feature, title=f'{feature} Distribution by Segment')
            st.plotly_chart(fig)

        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        correlation_matrix = df[selected_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(fig)

        # Interactive prediction
        st.subheader("Analyze New Customer")
        input_values = {}
        for feature in selected_features:
            if feature == 'Annual_Income':
                input_values[feature] = st.number_input(feature, min_value=0.0, max_value=200000.0, value=50000.0)
            elif feature == 'Purchase_Frequency':
                input_values[feature] = st.number_input(feature, min_value=0.0, max_value=50.0, value=10.0)
            elif feature == 'Days_Since_Last_Purchase':
                input_values[feature] = st.number_input(feature, min_value=0.0, max_value=365.0, value=30.0)
            elif feature == 'Average_Order_Value':
                input_values[feature] = st.number_input(feature, min_value=0.0, max_value=1000.0, value=100.0)
            elif feature == 'Website_Time_Spent':
                input_values[feature] = st.number_input(feature, min_value=0.0, max_value=120.0, value=30.0)
            elif feature == 'App_Usage_Frequency':
                input_values[feature] = st.number_input(feature, min_value=0.0, max_value=50.0, value=10.0)

        if st.button("Analyze Customer"):
            # Create input array with only the selected features
            new_customer = np.array([[input_values[feature] for feature in selected_features]])
            new_customer_scaled = scaler.transform(new_customer)
            
            # Calculate membership values
            distances = cdist(new_customer_scaled, centers)
            membership = 1 / (distances ** (2/(fuzziness-1)))
            membership = membership / membership.sum(axis=1)[:, np.newaxis]
            
            # Get prediction
            prediction = np.argmax(membership)
            segment = cluster_labels[prediction]
            
            # Display results
            st.success(f"Customer belongs to: {segment}")
            st.write("Membership values:")
            for i in range(n_clusters):
                st.write(f"Segment {i+1}: {membership[0][i]:.2%}")

if __name__ == "__main__":
    run() 