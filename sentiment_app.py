import streamlit as st
import joblib
import json
from google.oauth2 import service_account
from google.cloud import translate_v2 as translate
import time
import os

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon=":movie_camera:", layout="centered")

st.markdown(
    """
    <style>
    /* Custom CSS for the app */
    section.main > div:first-child {
        padding-top: 1px;
    }
    h1 {
        margin-top: 0; 
        margin-bottom: 0.2em;
    }
    h2 {
        margin-top: 0;
        margin-bottom: 0.5em;
    }
    .stApp {
        background: linear-gradient(135deg, #240046, #5a189a);
        color: white;
        font-family: 'Georgia', serif;
    }
    textarea {
        background-color: rgb(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
        border: 1px solid #ffffff40;
        font-size: 16px;
    }
    textarea:focus {
        outline: none;
        border: 1px solid #ffffff80;
    }
    div.stButton > button {
        transition: none !important;
        border: 2px solid #111 !important;
        background-color: #111 !important;
        color: white !important;
        font-weight: normal !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover {
        background-color: #111 !important;
        color: white !important;
        border: 2px solid #111 !important;
    }
    div.stTextArea label {
        color: #ffffcc !important;
        font-weight: bold !important;
        font-size: 18px !important;
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown(
    """
    <h1 style="text-align: center; font-family: 'Georgia', serif; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
        Movie Review Sentiment Analyzer
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h2 style="text-align: center; font-size: 1.10em; font-family: 'Georgia', serif; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
        Enter a movie review (in any language) and get a sentiment prediction!
    </h2>
    """,
    unsafe_allow_html=True
)

# Translation credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "movie-review-translator-07db7e662290.json" 

# Load model and vectorizer
model = joblib.load('best_nb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Google translate client
translate_client = translate.Client()

# Randy's function to translate text if needed
def translate_if_needed(text):
    result = translate_client.translate(text, target_language="en")
    detected = result["detectedSourceLanguage"]
    translated_text = result["translatedText"]

    return detected, translated_text

# Function to classify the review
def classify_review(text):
    detected_lang, translated = translate_if_needed(text)
    vectorized = vectorizer.transform([translated])
    prediction = model.predict(vectorized)
    return detected_lang, translated, prediction[0]

if "review" not in st.session_state:
    st.session_state.review = ""

# Text area for user input
review = st.text_area("Movie Review", value=st.session_state.review, key="review", height=150, placeholder="Enter your review here...")

# Function to clear text area
def clear_text():
    st.session_state.review = ""

# Button to clear text area
col1, col2 = st.columns([8, 2])
with col2:
    st.button("Clear", on_click=clear_text)

# Button to predict sentiment
if st.button ("🎬 Predict Sentiment", use_container_width=True):
    if review.strip() == "":
        st.warning("Please enter a review:")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.3)
            detected_lang, translation, prediction = classify_review(review)
            # Retrieve full language name for display
            language_names = {
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ja": "Japanese",
                "zh": "Chinese",
                "ko": "Korean",
                "ru": "Russian",
                "ar": "Arabic",
                "hi": "Hindi",
                "bn": "Bengali",
                "tr": "Turkish"
            }
            lang_display = language_names.get(detected_lang.lower(), detected_lang.upper())

            if detected_lang.lower() != "en":

                # Show the translated text
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, #3a0ca3, #7209b7); padding: 20px; border-radius: 15px; margin-top: 20px; margin-bottom: 20px;'>
                        <h3 style='text-align: center; color: #ffffff;'>Translated from {lang_display}</h3>
                        <p style='text-align: center; font-weight: bold; font-size: 18px; color: #eeeeee; margin-top: 10px;'>Translated sentence: {translation}</p>
                        <br>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Using Rania's logic
            if prediction == 1:
                st.success("✅ It's a **positive** review!")
            else:
                st.error("❌ It's a **negative** review!")