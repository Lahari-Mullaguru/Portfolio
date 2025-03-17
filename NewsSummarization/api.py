from flask import Flask, request, jsonify
from utils import fetch_news, analyze_sentiment, text_to_speech, compare_sentiments

app = Flask(__name__)

@app.route('/fetch_news', methods=['POST'])
def fetch_news_api():
    data = request.json
    company = data.get("company", "")
    articles = fetch_news(company)
    
    sentiment_data, processed_articles = compare_sentiments(articles)

    return jsonify({
        "articles": processed_articles,
        "sentiment_analysis": sentiment_data
    })

@app.route('/tts', methods=['POST'])
def tts_api():
    data = request.json
    text = data.get("text", "")
    audio_file = text_to_speech(text)
    
    return jsonify({"audio_file": audio_file})

if __name__ == "__main__":
    app.run(debug=True)
