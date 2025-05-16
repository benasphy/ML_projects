import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

def generate_sample_data(n_transactions=1000):
    # Define product categories and their items with common combinations
    categories = {
        'Dairy': ['Milk', 'Cheese', 'Yogurt', 'Butter', 'Cream'],
        'Bakery': ['Bread', 'Cake', 'Cookies', 'Muffins', 'Pastries'],
        'Produce': ['Apples', 'Bananas', 'Oranges', 'Tomatoes', 'Lettuce'],
        'Meat': ['Chicken', 'Beef', 'Pork', 'Fish', 'Sausage'],
        'Snacks': ['Chips', 'Crackers', 'Nuts', 'Popcorn', 'Candy']
    }
    
    # Define common item combinations
    common_combinations = [
        ['Milk', 'Bread'],
        ['Cheese', 'Bread'],
        ['Chicken', 'Lettuce'],
        ['Chips', 'Soda'],
        ['Cookies', 'Milk'],
        ['Apples', 'Bananas'],
        ['Beef', 'Tomatoes'],
        ['Crackers', 'Cheese']
    ]
    
    # Generate transactions
    transactions = []
    for _ in range(n_transactions):
        transaction = []
        
        # 70% chance to include a common combination
        if np.random.random() < 0.7:
            # Randomly select a combination index
            combo_idx = np.random.randint(0, len(common_combinations))
            transaction.extend(common_combinations[combo_idx])
        
        # Add 1-3 random items
        n_additional = np.random.randint(1, 4)
        for _ in range(n_additional):
            category = np.random.choice(list(categories.keys()))
            item = np.random.choice(categories[category])
            if item not in transaction:  # Avoid duplicates
                transaction.append(item)
        
        transactions.append(transaction)
    
    return transactions

def format_rule(rule):
    """Convert frozenset to string for display"""
    antecedents = ', '.join(list(rule['antecedents']))
    consequents = ', '.join(list(rule['consequents']))
    return f"{antecedents} → {consequents}"

def run():
    st.header("Market Basket Analysis")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/Association_Rule_Learning)", unsafe_allow_html=True)

    # Load or generate dataset
    uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Convert to list of transactions
        transactions = df.values.tolist()
    else:
        st.info("Using sample transaction data")
        transactions = generate_sample_data()

    # Convert transactions to one-hot encoded DataFrame
    unique_items = list(set(item for transaction in transactions for item in transaction))
    df = pd.DataFrame([[1 if item in transaction else 0 for item in unique_items] 
                      for transaction in transactions], columns=unique_items)

    # Display data info
    st.subheader("Dataset Information")
    st.write(f"Number of transactions: {len(transactions)}")
    st.write(f"Number of unique items: {len(unique_items)}")
    st.write("Sample transactions:")
    st.dataframe(df.head())

    # Parameters
    st.subheader("Association Rule Parameters")
    min_support = st.slider("Minimum Support", min_value=0.001, max_value=0.5, value=0.005, step=0.001)
    min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
    min_lift = st.slider("Minimum Lift", min_value=1.0, max_value=5.0, value=1.1, step=0.1)

    if st.button("Generate Rules"):
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
                
                # Network visualization
                st.subheader("Rule Network")
                G = nx.Graph()
                
                # Add nodes and edges
                for _, rule in rules.iterrows():
                    antecedents = list(rule['antecedents'])[0]
                    consequents = list(rule['consequents'])[0]
                    G.add_edge(antecedents, consequents, weight=rule['lift'])
                
                # Create plot
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                       node_size=1500, font_size=10, font_weight='bold')
                st.pyplot(plt)
                
                # Top rules by lift
                st.subheader("Top Rules by Lift")
                top_rules = rules.sort_values('lift', ascending=False).head(5)
                for _, rule in top_rules.iterrows():
                    antecedents = list(rule['antecedents'])[0]
                    consequents = list(rule['consequents'])[0]
                    st.write(f"If {antecedents} → {consequents}")
                    st.write(f"Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
                    st.write("---")
            else:
                st.warning("No rules found with the current parameters. Try adjusting the thresholds.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Try adjusting the parameters or using different data.")

if __name__ == "__main__":
    run() 