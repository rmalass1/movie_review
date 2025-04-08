import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)

#Load the two files where X represents the reviews and y represents the sentiment column
X = joblib.load('x_tfidf.pkl')
y = joblib.load('y_labels.pkl')

#Split the dataset into training and testing for model evaluation 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

#define the hyperparameters
param_grid = {'alpha': [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.21, 0.23, 0.3, 0.31, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.42, 0.8, 1.0]}

#Build a Naive Bayes Classifier 
model = MultinomialNB()

#fit the model
grid = GridSearchCV(estimator=model, param_grid= param_grid, scoring= 'f1', cv=8)
grid.fit(X_train, y_train)

#Get the best hyperparameter
print("\nBest Alpha: ", grid.best_params_['alpha'])

#Train the final best model
best_model = MultinomialNB(alpha= grid.best_params_['alpha'])
best_model.fit(X_train, y_train)

#Use the model to predict sentiment labels
y_pred = best_model.predict(X_test)

#Calculate the accuracy of the model
accuracy = accuracy_score(y_pred, y_test)

#Calculate the f1-score of the model
f1 = f1_score(y_pred, y_test)

#Show the confusion matrix
cm = confusion_matrix(y_test, y_pred)

#print the confusion matrix
print("\nConfusion Matrix:\n ", cm)

#print the evaluation matrics 
print("\nAccuracy: ", accuracy)
print("\nF1-score: ", f1)

#Save the best model
joblib.dump(best_model, 'best_nb_model.pkl')








