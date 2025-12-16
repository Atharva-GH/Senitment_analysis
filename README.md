# Sentiment Analysis using TF-IDF and Logistic Regression

This project is a multi-class sentiment (emotion) analysis system built using Natural Language Processing (NLP) techniques and a machine learning pipeline.  
It predicts the emotional tone of user-provided text and presents the result through an interactive Streamlit web application.

## Problem Statement

Understanding emotions in text is a key task in NLP, with applications in customer feedback analysis, social media monitoring, chatbots, and opinion mining.

The goal of this project is to classify text into one of six emotion categories using classical machine-learning techniques.

## Emotions Supported

| Label | Emotion |
|------|--------|
| 0 | Sadness 😢 |
| 1 | Anger 😠 |
| 2 | Love ❤️ |
| 3 | Surprise 😲 |
| 4 | Fear 😨 |
| 5 | Joy 😄 |

## Tech Stack

- Python
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Streamlit

## Machine Learning Pipeline

### Text Preprocessing
- Lowercasing
- Removal of punctuation
- Removal of numbers
- Tokenization
- Stopword removal

### Feature Extraction
- TF-IDF Vectorization

### Model Training
Three models were evaluated:
- Bag of Words + Naive Bayes
- TF-IDF + Naive Bayes
- TF-IDF + Logistic Regression (Best performing model)

Logistic Regression was selected based on superior accuracy.

## Streamlit Web Application

The project includes a professional Streamlit UI with:
- Sidebar navigation
- Emotion prediction with emojis
- Cleaned text preview
- Model information section

## Project Structure

sentiment-analysis-streamlit/
│── app.py
│── model.pkl
│── tfidf.pkl
│── label_map.pkl
│── requirements.txt
│── README.md

## How to Run Locally

1. Clone the repository or download the files  
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py
4. Open the provided local URL in your browser

## Deployment

This app can be deployed using Streamlit Cloud:
1. Upload the project to GitHub
2. Connect the repository to Streamlit Cloud
3. Select app.py as the entry point
4. Deploy

## Key Learnings

- NLP preprocessing techniques
- TF-IDF feature engineering
- Multi-class text classification
- Model comparison and selection
- Building and deploying ML apps with Streamlit

## Future Improvements

- Add prediction confidence scores
- Batch text classification
- Explore deep learning models
- Improve UI and UX

## Author

Atharva Gupta  
B.Tech Computer Science, KIIT University  

