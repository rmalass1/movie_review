import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('best_nb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# UI using streamlit
st.title("Movie Review Sentiment Analyzer")
st.write("Enter a movie review:")

# Text area for user input
review = st.text_area("Movie Review", height=150)

# Predict button
if st.button("Predict Statement"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Transform review and predict
        X_new = vectorizer.transform([review])
        prediction = model.predict(X_new)[0]
        if prediction == 1:
            st.success("It's a **positive** review!")
        else:
            st.error("It's a **negative** review!")