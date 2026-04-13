import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. The Data (Small sample for training)
data = {
    'text': [
        "unauthorized credit card charge", "late fee on my invoice", 
        "lost my credit card", "cannot log into my bank account",
        "stolen card reported", "wrong billing amount on statement"
    ],
    'label': [0, 1, 0, 2, 0, 1] # 0: Card, 1: Billing, 2: Login
}
df = pd.DataFrame(data)

# --- EXPERIMENT A: CountVectorizer (Simple Word Counts) ---
vectorizer_a = CountVectorizer()
X_a = vectorizer_a.fit_transform(df['text'])
model_a = MultinomialNB().fit(X_a, df['label'])
acc_a = accuracy_score(df['label'], model_a.predict(X_a))

# --- EXPERIMENT B: TF-IDF (Weighted Word Importance) ---
vectorizer_b = TfidfVectorizer()
X_b = vectorizer_b.fit_transform(df['text'])
model_b = MultinomialNB().fit(X_b, df['label'])
acc_b = accuracy_score(df['label'], model_b.predict(X_b))

print(f"Experiment A Accuracy: {acc_a:.2f}")
print(f"Experiment B Accuracy: {acc_b:.2f}")

if acc_b >= acc_a:
    print("Strategy B (TF-IDF) is the winner. Engineering the Vectorizer worked!")
    # Deploy Trigger Test 2026
