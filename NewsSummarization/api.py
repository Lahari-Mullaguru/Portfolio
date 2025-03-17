from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from gtts import gTTS

app = Flask(__name__)

# Function to fetch news articles
def fetch_news(company_name):
    search_url = f"https://news.google.com/search?q={company_name}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    articles = []
    for item in soup.select("article")[:10]:
        title = item.find("h3")
        link = item.find("a")
        if title and link:
            articles.append({
                "title": title.get_text(),
                "url": "https://news.google.com" + link['href'][1:]
            })
    
    return articles

@app.route('/fetch_news', methods=['POST'])
def fetch_news_api():
    data = request.json
    company = data.get("company")
    articles = fetch_news(company)
    return jsonify({"articles": articles})

# Function to perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_api():
    data = request.json
    articles = data.get("articles", [])
    
    for article in articles:
        article["sentiment"] = analyze_sentiment(article["title"])
    
    return jsonify({"articles": articles})

# Function to convert text to Hindi speech
@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get("text", "No text provided")
    
    tts = gTTS(text, lang="hi")
    tts.save("output.mp3")
    
    return jsonify({"audio_file": "output.mp3"})

if __name__ == "__main__":
    app.run(debug=True)
