import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.header("Customer Behavior Analysis using HDBSCAN")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/DBSCAN_HDBSCAN)", unsafe_allow_html=True)

    # Load or generate dataset
    uploaded_file = st.file_uploader("Upload a CSV file with customer behavior data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using sample customer behavior data")
        # Generate sample customer behavior data
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
    selected_features = st.multiselect("Select features for behavior analysis", features, 
                                     default=['Annual_Income', 'Purchase_Frequency', 'Days_Since_Last_Purchase'])

    if len(selected_features) >= 2:
        # Prepare data
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # HDBSCAN parameters
        st.subheader("HDBSCAN Parameters")
        min_cluster_size = st.slider("Minimum Cluster Size", min_value=5, max_value=50, value=15)
        min_samples = st.slider("Minimum Samples", min_value=1, max_value=20, value=5)

        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        df['Cluster'] = clusterer.fit_predict(X_scaled)
        
        # Label clusters
        df['Segment'] = df['Cluster'].apply(lambda x: f'Segment {x}' if x != -1 else 'Noise')

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
            prediction = clusterer.fit_predict(np.vstack([X_scaled, new_customer_scaled]))[-1]
            segment = f'Segment {prediction}' if prediction != -1 else 'Noise'
            st.success(f"Customer belongs to: {segment}")

if __name__ == "__main__":
    run() 