from flask import Flask, render_template, request
import pandas as pd
import os

from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect

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
# LANGUAGE DETECTION
# ==========================

def is_english(text):
    try:
        if len(text.split()) < 2:
            return True
        return detect(text) == "en"
    except:
        return True

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def preprocess_text(text):
    text = text.strip()

    if is_english(text):
        return text
    else:
        return translate_to_english(text)

# ==========================
# RULE-BASED TEXT CLASSIFICATION
# ==========================

def classify_text(text):

    text = text.lower()

    categories = []

    menu_keywords = [
    "coffee","latte","cappuccino","espresso","americano",
    "mocha","macchiato","tea","milk tea","green tea",
    "cake","cheesecake","brownie","croissant","muffin",
    "toast","sandwich","waffle","pastry","dessert",
    "ice cream","chocolate","food","drink","beverage",
    "menu","bread","pasta","burger","pizza"
]

    service_keywords = [
    "service","staff","waiter","waitress","barista",
    "cashier","manager","employee","crew",
    "served","serving","greet","greeting",
    "friendly","helpful","polite","rude",
    "slow","fast service","customer service"
]

    price_keywords = [
    "price","expensive","cheap","affordable",
    "overpriced","worth","value","cost","pricing"
]

    atmosphere_keywords = [
    "atmosphere","ambience","environment","vibe",
    "music","lighting","decor","decoration","interior",
    "design","aesthetic","cozy","cosy","comfortable",
    "quiet","noisy","crowded","peaceful",
    "clean","dirty","spacious","temperature"
]

    facility_keywords = [
    "wifi","internet","socket","plug",
    "parking","toilet","restroom",
    "facility","facilities",
    "charging","power outlet"
]

    if any(word in text for word in menu_keywords):
        categories.append("Menu")

    if any(word in text for word in service_keywords):
        categories.append("Service")

    if any(word in text for word in price_keywords):
        categories.append("Price")

    if any(word in text for word in atmosphere_keywords):
        categories.append("Environment")

    if any(word in text for word in facility_keywords):
        categories.append("Facilities")

    if not categories:
        categories.append("General")

    return ", ".join(categories)

# ==========================
# CREATE CSV
# ==========================

if not os.path.exists("submitted_reviews.csv"):

    pd.DataFrame(
        columns=["Review", "Sentiment", "Category"]
    ).to_csv(
        "submitted_reviews.csv",
        index=False
    )

# ==========================
# HOME PAGE
# ==========================

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        review = request.form["review"].strip()

        if review:

            translated_review = preprocess_text(review)

            sentiment = analyze_vader(translated_review)

            category = classify_text(translated_review)

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
        "index2.html",
        admin_data=admin_data.to_dict("records")
    )

# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":
    app.run(debug=True)