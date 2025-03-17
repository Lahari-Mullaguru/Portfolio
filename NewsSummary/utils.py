import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from gtts import gTTS
import os


def fetch_news(company_name):
    search_url = f"https://news.google.com/rss/search?q={company_name}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "xml")

    articles = []
    for item in soup.find_all("item")[:10]:  # Fetch top 10 articles
        title = item.title.text
        summary = item.description.text
        url = item.link.text

        articles.append({
            "title": title,
            "summary": summary,
            "url": url
        })

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
