# Mini Project:  Intelligent Multi-Class Natural Language
#         Text Sentiment Classifier
# -------------------------------------------------------
# Libraries used:
# pandas        -> dataset loading and handling
# numpy         -> numerical operations
# nltk          -> stopword removal and lemmatization
# scikit-learn  -> TF-IDF, train/test split, model, evaluation
# matplotlib/seaborn -> confusion matrix visualization
# -------------------------------------------------------

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
)

# -------------------------------------------------------
# 1. Download required NLTK resources
# -------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -------------------------------------------------------
# 2. Text preprocessing function
# -------------------------------------------------------
def preprocess_text(text):
    """Clean raw text: lowercase, remove punctuation, tokenize,
    remove stopwords, and lemmatize."""
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# -------------------------------------------------------
# 3. Load dataset
# -------------------------------------------------------
df = pd.read_csv(
    "twitter_training.csv",
    header= None,
    names=["tweet_id", "entity", "sentiment", "text"]
)
df = df[["text", "sentiment"]]
df = df.dropna(subset= ["text", "sentiment"])
df = df[df["sentiment"] != "Irrelevant"]


print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("Classes:")
for label in sorted(df["sentiment"].unique()):
    print(f"- {label}")
print()

# -------------------------------------------------------
# 4. Preprocess text column
# -------------------------------------------------------
df["clean_text"] = df["text"].apply(preprocess_text)

# -------------------------------------------------------
# 5. Train/test split
# -------------------------------------------------------
X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}\n")

# -------------------------------------------------------
# 6. TF-IDF vectorization (fit on train only, transform test)
# -------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------------------------------------
# 7. Train Logistic Regression model
# -------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("Model training completed.\n")

# -------------------------------------------------------
# 8. Evaluate model
# -------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print("===== MODEL EVALUATION =====\n")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}\n")

# -------------------------------------------------------
# 9. Confusion matrix
# -------------------------------------------------------
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -------------------------------------------------------
# 10. Custom user prediction
# -------------------------------------------------------
def predict_sentiment(raw_text):
    cleaned = preprocess_text(raw_text)
    vector = vectorizer.transform([cleaned])
    return model.predict(vector)[0]


print("===== SENTIMENT PREDICTION =====")
print("Enter a sentence to find out its predicted sentiment.\n")

while True:
    user_text = input("Enter your text: ")
    prediction = predict_sentiment(user_text)
    print(f"Predicted Sentiment: {prediction}\n")

    again = input("Enter another sentence? (yes/no): ").strip().lower()
    if again != "yes":
        print("Exiting program.")
        break
