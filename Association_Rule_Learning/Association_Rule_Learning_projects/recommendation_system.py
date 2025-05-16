import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

def generate_sample_data(n_users=1000):
    # Define product categories and their items
    categories = {
        'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Camera'],
        'Books': ['Fiction', 'Non-Fiction', 'Biography', 'Science', 'History'],
        'Movies': ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Documentary'],
        'Music': ['Pop', 'Rock', 'Classical', 'Jazz', 'Hip-Hop'],
        'Games': ['Action', 'Strategy', 'Puzzle', 'Sports', 'RPG']
    }
    
    # Define common user preferences
    common_preferences = [
        ['Smartphone', 'Headphones'],
        ['Laptop', 'Tablet'],
        ['Fiction', 'Biography'],
        ['Action', 'Sci-Fi'],
        ['Pop', 'Rock'],
        ['Action', 'Strategy'],
        ['Comedy', 'Drama'],
        ['Classical', 'Jazz']
    ]
    
    # Generate user interactions
    interactions = []
    for _ in range(n_users):
        user_interactions = []
        
        # 80% chance to include a common preference
        if np.random.random() < 0.8:
            pref_idx = np.random.randint(0, len(common_preferences))
            user_interactions.extend(common_preferences[pref_idx])
        
        # Add 2-4 random items
        n_additional = np.random.randint(2, 5)
        for _ in range(n_additional):
            category = np.random.choice(list(categories.keys()))
            item = np.random.choice(categories[category])
            if item not in user_interactions:  # Avoid duplicates
                user_interactions.append(item)
        
        interactions.append(user_interactions)
    
    return interactions

def format_rule(rule):
    """Convert frozenset to string for display"""
    antecedents = ', '.join(list(rule['antecedents']))
    consequents = ', '.join(list(rule['consequents']))
    return f"{antecedents} → {consequents}"

def run():
    st.header("Recommendation System using Association Rules")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/Association_Rule_Learning)", unsafe_allow_html=True)

    # Load or generate dataset
    uploaded_file = st.file_uploader("Upload a CSV file with user-item interactions", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Convert to list of interactions
        interactions = df.values.tolist()
    else:
        st.info("Using sample user-item interaction data")
        interactions = generate_sample_data()

    # Convert interactions to one-hot encoded DataFrame
    unique_items = list(set(item for interaction in interactions for item in interaction))
    df = pd.DataFrame([[1 if item in interaction else 0 for item in unique_items] 
                      for interaction in interactions], columns=unique_items)

    # Display data info
    st.subheader("Dataset Information")
    st.write(f"Number of users: {len(interactions)}")
    st.write(f"Number of unique items: {len(unique_items)}")
    st.write("Sample interactions:")
    st.dataframe(df.head())

    # Parameters
    st.subheader("Association Rule Parameters")
    min_support = st.slider("Minimum Support", min_value=0.001, max_value=0.5, value=0.003, step=0.001)
    min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.15, step=0.05)
    min_lift = st.slider("Minimum Lift", min_value=1.0, max_value=5.0, value=1.1, step=0.1)

    if st.button("Generate Recommendations"):
        try:
            # Generate frequent itemsets
            frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                st.warning("No frequent itemsets found. Try lowering the minimum support threshold.")
                return
            
            # Generate rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            rules = rules[rules['lift'] >= min_lift]
            
            if len(rules) > 0:
                # Create a copy of rules with formatted strings for display
                display_rules = rules.copy()
                display_rules['rule'] = display_rules.apply(format_rule, axis=1)
                
                # Display rules
                st.subheader("Association Rules")
                st.dataframe(display_rules[['rule', 'support', 'confidence', 'lift']])
                
                # Visualize support vs confidence
                fig = px.scatter(rules, x="support", y="confidence", 
                               size="lift", color="lift",
                               hover_data=["antecedents", "consequents"],
                               title="Support vs Confidence")
                st.plotly_chart(fig)
                
                # Interactive recommendation
                st.subheader("Get Recommendations")
                selected_items = st.multiselect("Select items you like:", unique_items)
                
                if selected_items:
                    # Find rules where selected items are in antecedents
                    recommendations = []
                    for _, rule in rules.iterrows():
                        if all(item in rule['antecedents'] for item in selected_items):
                            recommendations.extend(list(rule['consequents']))
                    
                    if recommendations:
                        # Remove duplicates and selected items
                        recommendations = list(set(recommendations) - set(selected_items))
                        
                        # Sort by frequency
                        recommendation_counts = pd.Series(recommendations).value_counts()
                        
                        st.write("Recommended items based on your selection:")
                        for item, count in recommendation_counts.items():
                            st.write(f"- {item} (recommended {count} times)")
                    else:
                        st.info("No specific recommendations found. Try selecting different items or adjusting the parameters.")
            else:
                st.warning("No rules found with the current parameters. Try adjusting the thresholds.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Try adjusting the parameters or using different data.")

if __name__ == "__main__":
    run() 