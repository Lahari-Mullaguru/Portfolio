import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from gtts import gTTS
import os

# Function to extract news articles
def fetch_news(company_name):
    search_url = f"https://news.google.com/search?q={company_name}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = []
    for item in soup.select("article")[:10]:  # Fetching top 10 articles
        title = item.find("h3")
        link = item.find("a")
        if title and link:
            articles.append({
                "title": title.get_text(),
                "url": "https://news.google.com" + link['href'][1:]
            })

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
