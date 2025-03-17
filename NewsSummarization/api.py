from fastapi import FastAPI
import requests
from textblob import TextBlob
import gtts
from pydantic import BaseModel
import os

app = FastAPI()

NEWS_API_KEY = "ed699af727aa4c0d9463f19babe2fb7e"
NEWS_API_URL = "https://newsapi.org/v2/everything"

class CompanyRequest(BaseModel):
    company: str

def fetch_news(company):
    """Fetch news articles related to the company"""
    params = {
        "q": company,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 10
    }
    response = requests.get(NEWS_API_URL, params=params)
    data = response.json()
    articles = data.get("articles", [])
    
    news_list = []
    for article in articles:
        news_list.append({
            "title": article["title"],
            "summary": article["description"] if article["description"] else "No summary available",
            "content": article["content"] if article["content"] else "No content available"
        })
    
    return news_list

def analyze_sentiment(text):
    """Perform sentiment analysis"""
    analysis = TextBlob(text)
    sentiment = "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"
    return sentiment

def generate_hindi_tts(text, filename="output.mp3"):
    """Convert summary to Hindi speech"""
    tts = gtts.gTTS(text, lang="hi")
    tts.save(filename)
    return filename

@app.post("/get_news/")
async def get_news(request: CompanyRequest):
    """Fetch news, perform sentiment analysis, and generate Hindi speech"""
    company = request.company
    articles = fetch_news(company)
    
    if not articles:
        return {"error": "No articles found"}

    sentiment_summary = []
    full_text = ""
    
    for article in articles:
        sentiment = analyze_sentiment(article["content"])
        article["sentiment"] = sentiment
        sentiment_summary.append(sentiment)
        full_text += article["summary"] + " "

    # Generate Hindi TTS
    tts_filename = generate_hindi_tts(full_text)

    # Sentiment distribution
    sentiment_counts = {
        "Positive": sentiment_summary.count("Positive"),
        "Negative": sentiment_summary.count("Negative"),
        "Neutral": sentiment_summary.count("Neutral"),
    }

    return {
        "company": company,
        "articles": articles,
        "sentiment_summary": sentiment_counts,
        "audio": tts_filename
    }
