import joblib

#Load the best model and the vectorizer
model = joblib.load('best_nb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

#Input new review for prediction
review = input("Enter a movie review: ")

#Transform the review into numerical values
X_new = vectorizer.transform([review])

#Make a prediction for the sentiment of the review
predict_s = model.predict(X_new)[0]

#Analyze the prediction result
if predict_s == 1: 
    print("It's a positive review")
else:
    print("It's a negative review")