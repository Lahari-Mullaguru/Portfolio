import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from gtts import gTTS
import os

def fetch_news(company_name):
    API_KEY = "ed699af727aa4c0d9463f19babe2fb7e"  # Your API Key
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    # Print API Response for Debugging
    print("Full API Response in Flask API:", data)

    articles = []
    if "articles" in data and len(data["articles"]) > 0:
        for item in data["articles"][:10]:  # Fetch first 10 articles
            articles.append({
                "title": item.get("title", "No title available"),
                "summary": item.get("description", "No description available"),
                "url": item.get("url", ""),
                "source": item["source"]["name"]
            })
    else:
        print(f"Warning: No articles found for {company_name}")

    return articles



def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

def compare_sentiments(articles):
    sentiment_count = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for article in articles:
        sentiment = analyze_sentiment(article["summary"])
        article["sentiment"] = sentiment
        sentiment_count[sentiment] += 1

    return sentiment_count, articles

def text_to_speech(text):
    tts = gTTS(text, lang="hi")
    file_path = "static/output.mp3"
    tts.save(file_path)
    return file_path
