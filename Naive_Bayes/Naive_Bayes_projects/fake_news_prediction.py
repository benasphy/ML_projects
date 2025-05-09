import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
    st.header("Fake News Prediction")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Naive_Bayes)", unsafe_allow_html=True)

    # Load dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X = df[['title', 'news_url', 'source_domain', 'tweet_num']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        y = df['real']
    else:
        st.info("Using default dataset: FakeNewsNet.csv")
        df = pd.read_csv("Naive_Bayes/Naive_Bayes_projects/FakeNewsNet.csv")
        X = df[['title', 'news_url', 'source_domain', 'tweet_num']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        y = df['real']

    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Number of Articles:", len(df))
    with col2:
        class_dist = df['real'].value_counts()
        fig = px.pie(values=class_dist.values, names=['Fake', 'Real'],
                    title='News Distribution')
        st.plotly_chart(fig)

    # Text Analysis
    st.subheader("Text Analysis")
    
    # Preprocess text
    df['cleaned_text'] = X.apply(preprocess_text)
    
    # Text length analysis
    df['text_length'] = X.str.len()
    
    # Text length distribution
    fig = px.box(df, x='real', y='text_length',
                title='Text Length Distribution by Category',
                labels={'real': 'Category', 'text_length': 'Text Length'})
    st.plotly_chart(fig)

    # Vectorize text
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display metrics
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Real News Precision", 
                 f"{classification_report(y_test, y_pred, output_dict=True)['1']['precision']:.2%}")
    with col3:
        st.metric("Real News Recall",
                 f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.2%}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Fake', 'Real'],
                   y=['Fake', 'Real'],
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
        # Most indicative of real news
        fig = px.bar(feature_importance.head(10), x='Word', y='Importance',
                    title='Top Words Indicating Real News')
        st.plotly_chart(fig)
    
    with col2:
        # Most indicative of fake news
        fig = px.bar(feature_importance.tail(10), x='Word', y='Importance',
                    title='Top Words Indicating Fake News')
        st.plotly_chart(fig)

    # Predict custom input
    st.subheader("Test a News Article")
    news_text = st.text_area("Enter news article text:", height=200)
    
    if st.button("Check News"):
        if news_text:
            # Preprocess and vectorize input
            cleaned_input = preprocess_text(news_text)
            input_vectorized = vectorizer.transform([cleaned_input])
            
            # Make prediction
            prediction = model.predict(input_vectorized)[0]
            probabilities = model.predict_proba(input_vectorized)[0]
            
            # Display prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", "Real News" if prediction == 1 else "Fake News")
            with col2:
                st.metric("Confidence", f"{max(probabilities):.2%}")
            
            # Visualize prediction probabilities
            fig = go.Figure(data=[
                go.Bar(x=['Fake News', 'Real News'],
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
                st.write("**Text Length:**", len(news_text))
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
                        indicator = "→ Real News" if score > 0 else "→ Fake News"
                        st.write(f"- {word} {indicator}")

if __name__ == "__main__":
    run() 