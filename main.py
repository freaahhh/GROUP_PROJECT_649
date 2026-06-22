from flask import Flask, render_template, request
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from langdetect import detect

app = Flask(__name__)

# ==========================
# SENTIMENT ANALYSIS (VADER)
# ==========================

analyzer = SentimentIntensityAnalyzer()

def analyze_vader(text):
    score = analyzer.polarity_scores(text)['compound']

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# ==========================
# TRANSLATION + LANGUAGE CHECK
# ==========================

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def preprocess_text(text):
    text = text.strip()

    if is_english(text):
        return text
    else:
        return translate_to_english(text)

# ==========================
# TRAIN ML MODEL
# ==========================

df = pd.read_csv("cafe_customer_reviews.csv")

X = df["Review"]
y = df["Category"]

model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

model.fit(X, y)

def classify_text(text):
    return model.predict([text])[0]

# ==========================
# CREATE CSV IF NOT EXISTS
# ==========================

if not os.path.exists("submitted_reviews.csv"):
    pd.DataFrame(
        columns=["Review", "Sentiment", "Category"]
    ).to_csv("submitted_reviews.csv", index=False)

# ==========================
# HOME ROUTE
# ==========================

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        review = request.form["review"].strip()

        # preprocess (language handling)
        translated_review = preprocess_text(review)

        # NLP processing
        sentiment = analyze_vader(translated_review)
        category = classify_text(translated_review)

        # save to CSV
        new_row = pd.DataFrame({
            "Review": [review],
            "Sentiment": [sentiment],
            "Category": [category]
        })

        new_row.to_csv(
            "submitted_reviews.csv",
            mode="a",
            header=False,
            index=False,
            encoding="utf-8"
        )

    # admin dashboard data
    admin_data = pd.read_csv("submitted_reviews.csv")

    return render_template(
        "index.html",
        admin_data=admin_data.to_dict("records")
    )

# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":
    app.run(debug=True)