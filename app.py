import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup

# Initialize the Flask app
app = Flask(__name__)

# Load and prepare the data
data = pd.read_csv('fake_or_real_news.csv')
data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop('label', axis=1)

X, y = data['text'], data['fake']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

def get_article_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        return article_text
    except Exception as e:
        print(f"Error fetching article from URL: {e}")
        return None

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        article = file.read().decode('utf-8')
    elif 'text' in request.form and request.form['text'].strip() != '':
        article = request.form['text']
    elif 'url' in request.form and request.form['url'].strip() != '':
        url = request.form['url']
        article = get_article_text_from_url(url)
        if not article:
            return jsonify({'error': 'Could not retrieve article from URL'}), 400
    else:
        return jsonify({'error': 'No text provided'}), 400
    
    article_vectorized = vectorizer.transform([article])
    prediction = clf.predict(article_vectorized)
    result = 'REAL' if prediction[0] == 0 else 'FAKE'
    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(debug=True)
