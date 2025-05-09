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
    st.header("Loan Approval Analysis using K-Means Clustering")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/K-Means)", unsafe_allow_html=True)

    # Load or generate dataset
    uploaded_file = st.file_uploader("Upload a CSV file with loan data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using sample loan data")
        # Generate sample loan data
        np.random.seed(42)
        n_applicants = 1000
        
        data = {
            'Applicant_ID': range(1, n_applicants + 1),
            'Annual_Income': np.random.randint(30000, 120000, n_applicants),
            'Credit_Score': np.random.randint(300, 850, n_applicants),
            'Loan_Amount': np.random.randint(5000, 50000, n_applicants),
            'Debt_to_Income_Ratio': np.random.uniform(0.1, 0.5, n_applicants),
            'Employment_Years': np.random.randint(0, 30, n_applicants),
            'Age': np.random.randint(18, 65, n_applicants)
        }
        df = pd.DataFrame(data)

    # Display data info
    st.subheader("Dataset Information")
    st.write(f"Number of applicants: {len(df)}")
    st.write("Sample data:")
    st.dataframe(df.head())

    # Feature selection
    st.subheader("Feature Selection")
    features = ['Annual_Income', 'Credit_Score', 'Loan_Amount', 'Debt_to_Income_Ratio', 
                'Employment_Years', 'Age']
    selected_features = st.multiselect("Select features for clustering", features, 
                                     default=['Annual_Income', 'Credit_Score', 'Loan_Amount', 'Debt_to_Income_Ratio'])

    if len(selected_features) >= 2:
        # Prepare data
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means parameters
        st.subheader("Clustering Parameters")
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=6, value=3)
        random_state = st.slider("Random State", min_value=0, max_value=100, value=42)

        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        df['Risk_Category'] = kmeans.fit_predict(X_scaled)

        # Label clusters based on average credit score and income
        cluster_means = df.groupby('Risk_Category')[['Credit_Score', 'Annual_Income']].mean()
        risk_order = cluster_means['Credit_Score'].rank(ascending=False).astype(int) - 1
        risk_labels = {i: f"{['Low', 'Medium', 'High'][j]} Risk" 
                      for i, j in risk_order.items()}
        df['Risk_Category'] = df['Risk_Category'].map(risk_labels)

        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['PCA1'] = X_pca[:, 0]
        df['PCA2'] = X_pca[:, 1]

        # PCA Scatter plot
        fig = px.scatter(df, x='PCA1', y='PCA2', color='Risk_Category',
                        hover_data=selected_features,
                        title='Loan Approval Clusters (PCA Visualization)')
        st.plotly_chart(fig)

        # Cluster Analysis
        st.subheader("Cluster Analysis")
        for risk in df['Risk_Category'].unique():
            st.write(f"\n{risk} Applicants:")
            cluster_data = df[df['Risk_Category'] == risk]
            st.write(f"Number of applicants: {len(cluster_data)}")
            
            # Display cluster statistics
            stats = cluster_data[selected_features].describe()
            st.write("Cluster Statistics:")
            st.dataframe(stats)

        # Feature importance visualization
        st.subheader("Feature Importance by Cluster")
        cluster_means = df.groupby('Risk_Category')[selected_features].mean()
        fig = px.bar(cluster_means, title='Average Feature Values by Risk Category',
                    labels={'value': 'Average Value', 'variable': 'Feature'})
        st.plotly_chart(fig)

        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        correlation_matrix = df[selected_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(fig)

        # Interactive prediction
        st.subheader("Predict Risk Category for New Applicant")
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Annual Income", min_value=30000, max_value=120000, value=50000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        with col2:
            loan_amount = st.number_input("Loan Amount", min_value=5000, max_value=50000, value=20000)
            dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.1, max_value=0.5, value=0.3)

        if st.button("Predict Risk Category"):
            new_applicant = np.array([[income, credit_score, loan_amount, dti_ratio]])
            new_applicant_scaled = scaler.transform(new_applicant)
            prediction = kmeans.predict(new_applicant_scaled)[0]
            risk_category = risk_labels[prediction]
            st.success(f"Predicted Risk Category: {risk_category}")

if __name__ == "__main__":
    run() 