import streamlit as st
import pickle
import nltk
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ------------------------------------------------------
# NLTK Resources
# ------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

# ------------------------------------------------------
# Page Configuration
# ------------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# ------------------------------------------------------
# Load Model & Artifacts
# ------------------------------------------------------
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
label_map = pickle.load(open("label_map.pkl", "rb"))

reverse_label_map = {v: k for k, v in label_map.items()}

# ------------------------------------------------------
# Emoji Mapping
# ------------------------------------------------------
emoji_map = {
    "sadness": "😢",
    "anger": "😠",
    "love": "❤️",
    "suprise": "😲",
    "fear": "😨",
    "joy": "😄"
}

# ------------------------------------------------------
# Preprocessing Function
# ------------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ------------------------------------------------------
# Navigation
# ------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["About", "Analyze Text", "Model Info"])

# ------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------
if page == "About":
    st.title("Sentiment Analysis using TF-IDF + Logistic Regression")

    st.write("""
    This application performs **emotion detection** on user-provided text using
    a machine learning pipeline built with:

    - NLP preprocessing
    - TF-IDF Vectorization
    - Logistic Regression classifier

    **Supported Emotions:**
    😢 Sadness | 😠 Anger | ❤️ Love | 😲 Surprise | 😨 Fear | 😄 Joy
    """)

# ------------------------------------------------------
# ANALYZE TEXT PAGE
# ------------------------------------------------------
elif page == "Analyze Text":
    st.title("Analyze Text Emotion")

    user_input = st.text_area("Enter text to analyze:", height=150)

    if st.button("Predict Emotion"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            cleaned_text = preprocess_text(user_input)
            vectorized_text = tfidf.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]
            emotion = reverse_label_map[prediction]
            emoji = emoji_map.get(emotion, "")

            st.subheader("Prediction Result")
            st.success(f"Detected Emotion: **{emotion.upper()}** {emoji}")

            st.subheader("Processed Text")
            st.write(cleaned_text)

# ------------------------------------------------------
# MODEL INFO PAGE
# ------------------------------------------------------
elif page == "Model Info":
    st.title("Model Information")

    st.write("""
    **Model:** Logistic Regression  
    **Vectorizer:** TF-IDF  

    **Emotion Labels:**
    - 0 → Sadness 😢  
    - 1 → Anger 😠  
    - 2 → Love ❤️  
    - 3 → Surprise 😲  
    - 4 → Fear 😨  
    - 5 → Joy 😄  

    This model was selected based on
    superior performance compared to other tested models.
    """)
