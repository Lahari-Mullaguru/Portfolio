from flask import Flask, request, jsonify
from utils import fetch_news

app = Flask(__name__)

@app.route('/fetch_news', methods=['POST'])
def fetch_news_api():
    data = request.json
    print(f"🔹 Received Request: {data}")  # Print request data for debugging

    company = data.get("company", "")
    articles = fetch_news(company)

    print(f"🔹 News Articles Extracted: {articles}")  # Print extracted articles

    return jsonify({"articles": articles})

if __name__ == "__main__":
    app.run(debug=True)
