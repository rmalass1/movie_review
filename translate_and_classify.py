from google.cloud import translate_v2 as translate
import joblib
import os

# Point to JSON API key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "movie-review-translator-07db7e662290.json"  

# Load model and vectorizer
model = joblib.load("best_nb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize translator
translate_client = translate.Client()

# Translate if needed
def translate_if_needed(text):
    result = translate_client.translate(text, target_language="en")
    detected = result["detectedSourceLanguage"]
    translated_text = result["translatedText"]

    if detected != "en":
        print(f"[Translated from {detected.upper()}] → {translated_text}")

    return translated_text

# Classify the review
def classify_review(text):
    translated = translate_if_needed(text)
    vectorized = vectorizer.transform([translated])
    prediction = model.predict(vectorized)
    return "Positive" if prediction[0] == 1 else "Negative"

# Get user input and run
if __name__ == "__main__":
    review = input("Enter a movie review: ")
    sentiment = classify_review(review)
    print(f"\nPredicted Sentiment: {sentiment}")