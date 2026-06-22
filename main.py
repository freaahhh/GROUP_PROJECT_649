from flask import Flask, render_template, request
import pandas as pd
import os

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# ==========================
# SENTIMENT ANALYSIS
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
# HOME PAGE
# ==========================

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        review = request.form["review"]

        sentiment = analyze_vader(review)
        category = classify_text(review)

        new_row = pd.DataFrame({
            "Review": [review],
            "Sentiment": [sentiment],
            "Category": [category]
        })

        new_row.to_csv(
            "submitted_reviews.csv",
            mode="a",
            header=False,
            index=False
        )

    admin_data = pd.read_csv("submitted_reviews.csv")

    return render_template(
        "index.html",
        admin_data=admin_data.to_dict("records")
    )

if __name__ == "__main__":
    app.run(debug=True)