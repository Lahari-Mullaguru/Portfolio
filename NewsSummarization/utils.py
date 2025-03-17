import requests
from textblob import TextBlob
import gtts

NEWS_API_KEY = "ed699af727aa4c0d9463f19babe2fb7e"
NEWS_API_URL = "https://newsapi.org/v2/everything"

def fetch_news(company):
    params = {
        "q": company,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 10
    }
    response = requests.get(NEWS_API_URL, params=params)
    return response.json().get("articles", [])

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

def generate_hindi_tts(text, filename="output.mp3"):
    tts = gtts.gTTS(text, lang="hi")
    tts.save(filename)
    return filename
