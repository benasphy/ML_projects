import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def preprocess_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_wordcloud(texts, title):
    """Generate and display a word cloud."""
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         min_font_size=10).generate(' '.join(texts))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud)
    ax.axis('off')
    ax.set_title(title)
    return fig

def run():
    st.header("Spam Detection using Naive Bayes")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Naive_Bayes)", unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Use default dataset
        df = pd.read_csv('Naive_Bayes/Naive_Bayes_projects/spam.csv', encoding='latin-1')
        st.info("Using default spam dataset")
    
    # Rename columns if needed
    if 'v1' in df.columns and 'v2' in df.columns:
        df.columns = ['Category', 'Message'] + list(df.columns[2:])
    
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Number of Messages:", len(df))
    with col2:
        class_dist = df['Category'].value_counts()
        fig = px.pie(values=class_dist.values, names=class_dist.index,
                    title='Message Distribution')
        st.plotly_chart(fig)
    
    # Text Analysis
    st.subheader("Text Analysis")
    
    # Preprocess messages
    df['cleaned_message'] = df['Message'].apply(preprocess_text)
    
    # Message length analysis
    df['message_length'] = df['Message'].str.len()
    
    # Message length distribution
    fig = px.box(df, x='Category', y='message_length',
                title='Message Length Distribution by Category')
    st.plotly_chart(fig)
    
    # Word count analysis
    df['word_count'] = df['cleaned_message'].str.split().str.len()
    fig = px.box(df, x='Category', y='word_count',
                title='Word Count Distribution by Category')
    st.plotly_chart(fig)
    
    # Word clouds
    st.subheader("Word Clouds")
    col1, col2 = st.columns(2)
    
    with col1:
        spam_texts = df[df['Category'] == 'spam']['cleaned_message']
        fig_spam = generate_wordcloud(spam_texts, 'Spam Messages')
        st.pyplot(fig_spam)
        
    with col2:
        ham_texts = df[df['Category'] == 'ham']['cleaned_message']
        fig_ham = generate_wordcloud(ham_texts, 'Ham Messages')
        st.pyplot(fig_ham)
    
    # Common words analysis
    st.subheader("Most Common Words")
    
    def get_common_words(texts, n=10):
        words = ' '.join(texts).split()
        return pd.DataFrame(Counter(words).most_common(n), columns=['Word', 'Count'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        spam_common = get_common_words(df[df['Category'] == 'spam']['cleaned_message'])
        fig = px.bar(spam_common, x='Word', y='Count',
                    title='Most Common Words in Spam Messages')
        st.plotly_chart(fig)
        
    with col2:
        ham_common = get_common_words(df[df['Category'] == 'ham']['cleaned_message'])
        fig = px.bar(ham_common, x='Word', y='Count',
                    title='Most Common Words in Ham Messages')
        st.plotly_chart(fig)
    
    # Model Training
    st.subheader("Model Training")
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_message'])
    y = df['Category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Spam Precision", 
                 f"{classification_report(y_test, y_pred, output_dict=True)['spam']['precision']:.2%}")
    with col3:
        st.metric("Spam Recall",
                 f"{classification_report(y_test, y_pred, output_dict=True)['spam']['recall']:.2%}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Ham', 'Spam'],
                   y=['Ham', 'Spam'],
                   text_auto=True,
                   aspect="auto")
    st.plotly_chart(fig)
    
    # Feature Importance
    st.subheader("Most Important Words")
    feature_importance = pd.DataFrame({
        'Word': vectorizer.get_feature_names_out(),
        'Importance': model.feature_log_prob_[1] - model.feature_log_prob_[0]
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        # Most indicative of spam
        fig = px.bar(feature_importance.head(10), x='Word', y='Importance',
                    title='Top Words Indicating Spam')
        st.plotly_chart(fig)
    
    with col2:
        # Most indicative of ham
        fig = px.bar(feature_importance.tail(10), x='Word', y='Importance',
                    title='Top Words Indicating Ham')
        st.plotly_chart(fig)
    
    # Interactive Prediction
    st.subheader("Spam Detection")
    user_input = st.text_area("Enter a message to classify:", height=100)
    
    if st.button("Detect Spam"):
        if user_input:
            # Preprocess and vectorize input
            cleaned_input = preprocess_text(user_input)
            input_vectorized = vectorizer.transform([cleaned_input])
            
            # Make prediction
            prediction = model.predict(input_vectorized)[0]
            probabilities = model.predict_proba(input_vectorized)[0]
            
            # Display prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", "Spam" if prediction == "spam" else "Ham")
            with col2:
                st.metric("Confidence", f"{max(probabilities):.2%}")
            
            # Visualize prediction probabilities
            fig = go.Figure(data=[
                go.Bar(x=['Ham', 'Spam'],
                      y=probabilities,
                      text=[f'{p:.2%}' for p in probabilities],
                      textposition='auto',
                )
            ])
            fig.update_layout(title='Prediction Probabilities',
                            xaxis_title='Category',
                            yaxis_title='Probability')
            st.plotly_chart(fig)
            
            # Message Analysis
            st.subheader("Message Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Message Length:**", len(user_input))
                st.write("**Word Count:**", len(cleaned_input.split()))
            with col2:
                # Get top contributing words
                words = cleaned_input.split()
                word_scores = []
                for word in set(words):
                    if word in vectorizer.vocabulary_:
                        idx = vectorizer.vocabulary_[word]
                        score = model.feature_log_prob_[1][idx] - model.feature_log_prob_[0][idx]
                        word_scores.append((word, score))
                
                if word_scores:
                    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                    st.write("**Top Contributing Words:**")
                    for word, score in word_scores[:5]:
                        indicator = "→ Spam" if score > 0 else "→ Ham"
                        st.write(f"- {word} {indicator}")

if __name__ == "__main__":
    run()