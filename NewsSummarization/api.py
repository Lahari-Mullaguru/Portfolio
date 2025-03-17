from flask import Flask, request, jsonify
from utils import fetch_news, analyze_sentiment, text_to_speech

app = Flask(__name__)

@app.route('/fetch_news', methods=['POST'])
def fetch_news_api():
    data = request.json
    company = data.get("company")
    articles = fetch_news(company)
    return jsonify({"articles": articles})

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_api():
    data = request.json
    articles = data.get("articles", [])
    
    for article in articles:
        article["sentiment"] = analyze_sentiment(article["title"])
    
    return jsonify({"articles": articles})

@app.route('/tts', methods=['POST'])
def tts_api():
    data = request.json
    text = data.get("text", "No text provided")
    audio_file = text_to_speech(text)
    return jsonify({"audio_file": audio_file})

if __name__ == "__main__":
    app.run(debug=True)
