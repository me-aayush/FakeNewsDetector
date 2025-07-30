from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

app = Flask(__name__)

# Load model and vectorizer
try:
    model = joblib.load('models/fake_news_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
except:
    print("Model files not found. Please run train_model.py first")
    exit()

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        clean_msg = clean_text(message)
        vectorized_msg = tfidf.transform([clean_msg]).toarray()
        prediction = model.predict(vectorized_msg)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)