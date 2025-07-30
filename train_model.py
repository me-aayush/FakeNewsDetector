import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
true_news = pd.read_csv('data/True.csv')
fake_news = pd.read_csv('data/Fake.csv')

# Add labels and combine
true_news['label'] = 1  # 1 for real news
fake_news['label'] = 0  # 0 for fake news
news_df = pd.concat([true_news, fake_news]).sample(frac=1).reset_index(drop=True)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Clean text
news_df['clean_text'] = news_df['text'].apply(clean_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(news_df['clean_text']).toarray()
y = news_df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model and vectorizer
joblib.dump(model, 'models/fake_news_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

print("Model training complete and saved to models/ directory")