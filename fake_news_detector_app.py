import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from transformers import pipeline

def preprocess_text(text):
    text = str(text).replace('\n', ' ').replace('\r', ' ')
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text.strip()

def chunk_text(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def summarize_long_text(text):
    summaries = []
    for chunk in chunk_text(text):
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

@st.cache_data
def train_model(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['Text'] = df['Text'].apply(preprocess_text)
    X = df['Text']
    y = df['Label']
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vect = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("### Model Evaluation:")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))
    joblib.dump(model, "fake_news_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    st.success("Model trained and saved successfully!")
    return model, vectorizer

@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def predict_news(text):
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)[0]
    return prediction

st.title("📢 Fake News Detector & Summarizer for Students")
st.markdown("This app helps students detect fake news and get concise summaries of news articles. Supports long articles and batch CSV analysis.")

csv_path = r"C:\Users\HP\Downloads\Fake_Real_Data.csv\Fake_Real_Data.csv"

option = st.radio("Do you want to train a new model or load existing?", ("Load Existing Model", "Train New Model"))

if option == "Train New Model":
    model, vectorizer = train_model(csv_path)
else:
    model, vectorizer = load_model()
    st.success("Model and vectorizer loaded successfully!")

input_option = st.radio("Choose input method:", ("Paste Article Text", "Upload CSV"))

if input_option == "Paste Article Text":
    article = st.text_area("Paste the news article here:", height=300)
    if st.button("Analyze Article"):
        if article.strip() != "":
            article_clean = preprocess_text(article)
            prediction = predict_news(article_clean)
            summary = summarize_long_text(article_clean)
            st.subheader("📰 Summary")
            st.write(summary)
            st.subheader("🔍 Prediction")
            st.write(f"**This news is likely:** {prediction}")
        else:
            st.warning("Please enter some text to analyze.")
else:
    uploaded_file = st.file_uploader("Upload CSV with 'Text' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        if 'Text' not in df.columns:
            st.error("CSV must have a 'Text' column.")
        else:
            st.success("CSV uploaded successfully!")
            st.write("Showing first 5 rows:")
            st.dataframe(df.head())
            if st.button("Analyze All Articles"):
                results = []
                for i, row in df.iterrows():
                    text = preprocess_text(row['Text'])
                    pred = predict_news(text)
                    summary = summarize_long_text(text)
                    results.append({"Text": text, "Prediction": pred, "Summary": summary})
                results_df = pd.DataFrame(results)
                st.write("✅ Analysis Complete")
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", data=csv, file_name="news_analysis_results.csv", mime="text/csv")
