# SMS Spam Classifier using Naive Bayes

This project builds an end-to-end spam detection model using the SMS Spam Collection dataset. It demonstrates how to preprocess raw text data, extract meaningful features, train a supervised model, evaluate it, and serialize the final pipeline.

Inspired by the [HTB Academy – AI Red Teamer](https://academy.hackthebox.com/module/292/) module, which covers real-world applications of AI in offensive and defensive security workflows.

## Problem Statement

Spam messages are unsolicited texts that often contain malicious links, scams, or misleading information. Detecting them reliably is critical for mobile communication security. The challenge is to build a machine learning model that can learn from labeled data (`ham` or `spam`) and make predictions on new messages.

## Dataset

We use the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), which contains:

- 5,574 SMS messages
- Two classes: `ham` (legitimate) and `spam` (unwanted)
- Collected from real sources including Grumbletext, NUS Corpus, and others

## Methodology

### 1. Data Loading
The dataset is downloaded and unzipped into the `dataset/` folder using a utility script.

### 2. Text Preprocessing
To convert messages into meaningful features:
- Lowercasing
- Removing special characters (except `$`, `!`)
- Tokenization using regex
- Removing English stopwords
- Stemming with PorterStemmer

### 3. Feature Extraction
We use CountVectorizer with:
- ngram_range=(1, 2) to include unigrams and bigrams
- min_df=1, max_df=0.9 to control word frequency filtering

### 4. Model Pipeline
We use a Scikit-learn Pipeline:

message (text) → CountVectorizer → Multinomial Naive Bayes

### 5. Hyperparameter Tuning
Using GridSearchCV to find the best alpha value for MultinomialNB, optimizing for F1-score.

### 6. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score for both classes
- Confusion Matrix

### 7. Model Saving
The trained model pipeline is saved with joblib as:

models/spam_model.joblib

### 8. Demo Predictions
We provide a few sample messages to show model output and prediction probabilities.

## Setup

### Clone the Repository

```bash
git clone https://github.com/yourusername/sms-spam-classifier.git
cd sms-spam-classifier
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Download the Dataset

```bash
python dataset/download.py
```

### Train the Model

```bash
python train_spam_classifier.py
```

## Example Output

```
[*] Best alpha: 0.25
[+] Accuracy: 98%
[+] F1-score (spam): 0.90

[+] Demo predictions:
"FREE iPad winner" → Spam (1.00)
"Hey, lunch today?" → Not-Spam (0.00)
```

## Project Structure

```
sms-spam-classifier/
├── dataset/
│   └── download.py            # Script to download and extract dataset
├── models/
│   └── spam_model.joblib      # Saved model pipeline
├── train_spam_classifier.py   # Main training + evaluation script
├── requirements.txt
└── README.md
```

## Reference to HTB Academy

This project aligns with the HTB Academy AI Red Teamer module, particularly:

- Section: Spam Classification with Naive Bayes
- Skills: Text preprocessing, Bag-of-Words, CountVectorizer, Pipelines, GridSearchCV
- Objective: Build and evaluate a spam detection model, and submit it for scoring

If you're enrolled in the module, this repo serves as a strong standalone companion project and submission reference.

## Requirements

```
pandas
scikit-learn
nltk
joblib
requests
```

Install via:

```bash
pip install -r requirements.txt
```

## License

MIT License — use it for learning, teaching, or extending into production tools.

## Credits

- Dataset by Tiago A. Almeida and José María Gómez Hidalgo  
- Project inspired by Hack The Box Academy

