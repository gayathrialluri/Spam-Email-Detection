import string

stop_words = {
    'a','an','the','and','or','is','are','was','were','to','of','in','on',
    'for','with','this','that','it','as','at','by','from','be','have','has'
}

import pandas as pd

# Load dataset
df = pd.read_csv("dataset/email_dataset/spam.csv")

# Rename columns
df.columns = ['label', 'message']

# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check result
print(df.head())
print(df['label'].value_counts())

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['message'] = df['message'].apply(preprocess_text)
print(df.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Features and labels
X = df['message']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

print(X.shape)

from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

from sklearn.naive_bayes import MultinomialNB

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training completed")

from sklearn.metrics import accuracy_score, classification_report

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

test_message = ["Congratulations! You have won a free prize. Call now."]
test_vector = vectorizer.transform(test_message)

prediction = model.predict(test_vector)

print("\nPrediction for test message:")
print("Spam" if prediction[0] == 1 else "Not Spam")
