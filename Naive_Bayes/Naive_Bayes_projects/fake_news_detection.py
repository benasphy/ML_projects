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
import seaborn as sns

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
    st.header("Fake News Detection using Naive Bayes")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Naive_Bayes)", unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Use default dataset
        df = pd.read_csv('Naive_Bayes/Naive_Bayes_projects/FakeNewsNet.csv')
        st.info("Using default FakeNewsNet dataset")
    
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Number of Articles:", len(df))
    with col2:
        class_dist = df['label'].value_counts()
        fig = px.pie(values=class_dist.values, names=['Real', 'Fake'],
                    title='News Distribution')
        st.plotly_chart(fig)
    
    # Text Analysis
    st.subheader("Text Analysis")
    
    # Preprocess text
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Text length analysis
    df['text_length'] = df['text'].str.len()
    
    # Text length distribution
    fig = px.box(df, x='label', y='text_length',
                title='Text Length Distribution by Category',
                labels={'label': 'Category', 'text_length': 'Text Length'})
    st.plotly_chart(fig)
    
    # Word count analysis
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    fig = px.box(df, x='label', y='word_count',
                title='Word Count Distribution by Category',
                labels={'label': 'Category', 'word_count': 'Word Count'})
    st.plotly_chart(fig)
    
    # Word clouds
    st.subheader("Word Clouds")
    col1, col2 = st.columns(2)
    
    with col1:
        fake_texts = df[df['label'] == 1]['cleaned_text']
        fig_fake = generate_wordcloud(fake_texts, 'Fake News')
        st.pyplot(fig_fake)
        
    with col2:
        real_texts = df[df['label'] == 0]['cleaned_text']
        fig_real = generate_wordcloud(real_texts, 'Real News')
        st.pyplot(fig_real)
    
    # Common words analysis
    st.subheader("Most Common Words")
    
    def get_common_words(texts, n=10):
        words = ' '.join(texts).split()
        return pd.DataFrame(Counter(words).most_common(n), columns=['Word', 'Count'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fake_common = get_common_words(df[df['label'] == 1]['cleaned_text'])
        fig = px.bar(fake_common, x='Word', y='Count',
                    title='Most Common Words in Fake News')
        st.plotly_chart(fig)
        
    with col2:
        real_common = get_common_words(df[df['label'] == 0]['cleaned_text'])
        fig = px.bar(real_common, x='Word', y='Count',
                    title='Most Common Words in Real News')
        st.plotly_chart(fig)
    
    # Model Training
    st.subheader("Model Training")
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    
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
        st.metric("Fake News Precision", 
                 f"{classification_report(y_test, y_pred, output_dict=True)['1']['precision']:.2%}")
    with col3:
        st.metric("Fake News Recall",
                 f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.2%}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Real', 'Fake'],
                   y=['Real', 'Fake'],
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
        # Most indicative of fake news
        fig = px.bar(feature_importance.head(10), x='Word', y='Importance',
                    title='Top Words Indicating Fake News')
        st.plotly_chart(fig)
    
    with col2:
        # Most indicative of real news
        fig = px.bar(feature_importance.tail(10), x='Word', y='Importance',
                    title='Top Words Indicating Real News')
        st.plotly_chart(fig)
    
    # Interactive Prediction
    st.subheader("Fake News Detection")
    user_input = st.text_area("Enter news text to classify:", height=200)
    
    if st.button("Detect Fake News"):
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
                st.metric("Prediction", "Fake News" if prediction == 1 else "Real News")
            with col2:
                st.metric("Confidence", f"{max(probabilities):.2%}")
            
            # Visualize prediction probabilities
            fig = go.Figure(data=[
                go.Bar(x=['Real News', 'Fake News'],
                      y=probabilities,
                      text=[f'{p:.2%}' for p in probabilities],
                      textposition='auto',
                )
            ])
            fig.update_layout(title='Prediction Probabilities',
                            xaxis_title='Category',
                            yaxis_title='Probability')
            st.plotly_chart(fig)
            
            # Text Analysis
            st.subheader("Text Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Text Length:**", len(user_input))
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
                        indicator = "→ Fake News" if score > 0 else "→ Real News"
                        st.write(f"- {word} {indicator}")

if __name__ == "__main__":
    run()