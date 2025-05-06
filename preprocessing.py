import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

#Download NLTK data
nltk.download('stopwords')
nltk.download('all')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

#Load the Dataset
movie = pd.read_csv("IMDB Dataset.csv")

#Convert Review column into lowercase
movie["review"] = movie["review"].str.lower()

#Remove HTML Tags
def remove_html_tags(text):
  return BeautifulSoup(text, "html.parser").get_text()

movie["review"] = movie["review"].apply(remove_html_tags)

#Remove Special Characters
def remove_special_characters(text):
  return re.sub(r"[^\w\s]", "", text)

movie["review"] = movie["review"].apply(remove_special_characters)

#Tokenize the Reviews
movie["review"] = movie["review"].str.split(" ")

#Remove Stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
  return [word for word in text if word not in stop_words]

movie["review"] = movie["review"].apply(remove_stopwords)

#Function to ensure the lemmetizer uses the correct part of speech for each word
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
#define the lemmetizer
lemmatizer = WordNetLemmatizer()
    
#Apply lemmatization using the get_wordnet_pos 
def lemmatize_passage(text):
  if isinstance(text, list):
        text = ' '.join(text)
  tokens = word_tokenize(text)
  tagged_tokens = pos_tag(tokens)
  lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens]
  return ' '.join(lemmatized)

movie["review"] = movie["review"].apply(lemmatize_passage)


#Convert the cleaned reviews into numerical vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(movie["review"])

#Convert the sentiment labels into numerical values (0 or 1)
le = LabelEncoder()
y = le.fit_transform(movie["sentiment"])

#Save the prprocessed data
joblib.dump(X, 'x_tfidf.pkl')
joblib.dump(y, 'y_labels.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

