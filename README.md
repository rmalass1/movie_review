# 🎥 Movie Review Sentiment Analysis

A multilingual sentiment analysis tool for classifying movie reviews as positive or negative. Built using a Naïve Bayes classifier trained on the IMDB dataset, it includes translation support via the Google Translate API and a user-friendly Streamlit interface.

## 🌟 Objective

To develop an accessible tool that predicts whether a movie review is positive or negative, even if it's written in a language other than English. The model uses a Naïve Bayes classifier trained on 50,000 labeled reviews and supports real-time predictions via a web interface.

## 📌 Problem Overview

With the vast number of movie reviews spread across platforms, it's difficult to manually evaluate sentiment. This project aims to simplify that by automatically classifying reviews using sentiment analysis. It also removes the language barrier by using translation so that users can write in any language.

## 🧠 Approach

- **Model**: Multinomial Naïve Bayes Classifier
- **Dataset**: IMDB dataset of 50,000 reviews labeled as positive or negative
- **Text Preprocessing**: Lowercasing, removing HTML tags, special characters, stopwords, lemmatization
- **Feature Extraction**: TF-IDF vectorization
- **Translation**: Google Cloud Translate API for multilingual support

## ⚙️ Software Architecture

- `preprocessing.py`: Cleans and preprocesses raw text data
- `movie.py`: Trains the Naïve Bayes model and evaluates performance
- `translate_and_classify.py`: Detects language and translates text to English
- `sentiment_app.py`: Streamlit UI for users to submit reviews and get predictions

### Saved Assets:
- `best_nb_model.pkl`: Trained classifier
- `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer
- `X_tfidf.pkl`, `y_labels.pkl`: Preprocessed data

## 📊 Evaluation Results

### Test Set:
- **Accuracy**: 86.37%
- **F1 Score**: 0.864

### Training Set:
- **Accuracy**: 93.85%
- **F1 Score**: 0.938

## 🚀 How to Run the Project

All commands should be run from the root project folder.

### ⚡ Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 🔄 To perform preprocessing

Run the following command to start the preprocessing:
```bash
python preprocessing.py
```

### 🔄 To train the model

Run the following command to start the training:
```bash
python movie.py
```

### 🔄 Launch the Application

Run the following command to start the Streamlit web app:
```bash
streamlit run sentiment_app.py
```

## 🚧 Future Improvements

- Improve translation accuracy and add sarcasm detection
- Expand the model to support neutral sentiment classification
- Explore alternative NLP models for comparison (e.g., Logistic Regression, SVM, LSTM)

---

Developed by Rania Malass, Randy Valdez, and Karene Rosales  
California State University Fullerton, Spring 2025
