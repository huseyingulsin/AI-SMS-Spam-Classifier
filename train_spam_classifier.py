# train_spam_classifier.py

import re
import joblib
import nltk
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# Download NLTK data
nltk.download("stopwords")

# Config
DATA_PATH = Path("dataset/SMSSpamCollection")
MODEL_PATH = Path("models/spam_model.joblib")
Path("models").mkdir(exist_ok=True)
Path("dataset").mkdir(exist_ok=True)

# Preprocessing tools
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
CLEAN_RE = re.compile(r"[^a-z\s$!]")


def preprocess(text):
    text = text.lower()
    text = CLEAN_RE.sub("", text)
    tokens = re.findall(r"\b\w+\b", text)
    tokens = [STEMMER.stem(w) for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)


def load_dataset():
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    df.drop_duplicates(inplace=True)
    df["message"] = df["message"].apply(preprocess)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def train_model(df):
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))),
        ("classifier", MultinomialNB())
    ])

    param_grid = {
        "classifier__alpha": [0.01, 0.1, 0.25, 0.5, 1.0]
    }

    grid = GridSearchCV(pipeline, param_grid, scoring="f1", cv=5)
    grid.fit(X_train, y_train)

    print("[*] Best alpha:", grid.best_params_["classifier__alpha"])

    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print(f"[+] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    if not DATA_PATH.exists():
        print("[!] Dataset not found. Run python dataset/download.py first.")
    else:
        df = load_dataset()
        train_model(df)
