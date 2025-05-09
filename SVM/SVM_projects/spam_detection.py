import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
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
    st.header("Spam Detection using SVM")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/SVM)", unsafe_allow_html=True)

    # Load dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using default dataset: spam.csv")
        df = pd.read_csv("Naive_Bayes/Naive_Bayes_projects/spam.csv", encoding='latin-1')

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

    # Data Analysis
    st.subheader("Data Analysis")
    
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

    # Data preprocessing
    X = df['Message']
    y = df['Category']
    
    # Convert labels to binary (spam = 1, ham = 0)
    y = (y == 'spam').astype(int)

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Model evaluation
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        cv_scores = cross_val_score(model, X_vec, y, cv=5)
        st.metric("Cross-validation Score", f"{cv_scores.mean():.2%}")
    with col3:
        st.metric("Cross-validation Std", f"{cv_scores.std():.2%}")

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
    # Convert sparse coefficients to dense array
    coef_dense = model.coef_[0].toarray().flatten()
    feature_importance = pd.DataFrame({
        'Word': vectorizer.get_feature_names_out(),
        'Importance': np.abs(coef_dense)
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

    # Prediction interface
    st.subheader("Test an Email")
    email_text = st.text_area("Enter email text:", height=200)
    
    if st.button("Check if Spam"):
        if email_text:
            # Preprocess and vectorize input
            cleaned_input = preprocess_text(email_text)
            email_vec = vectorizer.transform([cleaned_input])
            
            # Make prediction
            prediction = model.predict(email_vec)[0]
            probabilities = model.predict_proba(email_vec)[0]
            
            # Display prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", "Spam" if prediction == 1 else "Ham")
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
                st.write("**Message Length:**", len(email_text))
                st.write("**Word Count:**", len(cleaned_input.split()))
            with col2:
                # Get top contributing words
                words = cleaned_input.split()
                word_scores = []
                # Convert coefficients to dense array
                coef_dense = model.coef_[0].toarray().flatten()
                
                for word in set(words):
                    if word in vectorizer.vocabulary_:
                        idx = vectorizer.vocabulary_[word]
                        if idx < len(coef_dense):  # Ensure index is within bounds
                            score = coef_dense[idx]
                            word_scores.append((word, score))
                
                if word_scores:
                    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                    st.write("**Top Contributing Words:**")
                    for word, score in word_scores[:5]:
                        indicator = "→ Spam" if score > 0 else "→ Ham"
                        st.write(f"- {word} {indicator}")

if __name__ == "__main__":
    run() 