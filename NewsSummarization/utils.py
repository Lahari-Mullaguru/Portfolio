import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from gtts import gTTS
import os

# Function to extract news articles
import requests

def fetch_news(company_name):
    API_KEY = "ed699af727aa4c0d9463f19babe2fb7e"  # API Key
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    articles = []
    if "articles" in data and data["articles"]:  # Check if there are articles
        for item in data["articles"][:10]:  # Get only the first 10
            articles.append({
                "title": item.get("title", "No title available"),
                "summary": item.get("description", "No description available"),
                "url": item.get("url", ""),
                "source": item["source"]["name"]
            })
    else:
        print(f"Error: No articles found for {company_name}. Response: {data}")

    return articles

# Function for sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

# Function for text-to-speech
def text_to_speech(text):
    tts = gTTS(text, lang="hi")
    tts.save("static/output.mp3")
    return "static/output.mp3"
