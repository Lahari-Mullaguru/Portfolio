from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from utils import fetch_news, analyze_sentiment, generate_comparative_analysis, text_to_speech
import json

app = FastAPI()

# Serve static files from the "static" folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/analyze-news")
def analyze_news(company_name: str):
    # Fetch news articles
    articles = fetch_news(company_name)
    
    if not articles:
        return {"error": "No articles found for the given company."}
    
    # Perform sentiment analysis
    articles_with_sentiment = analyze_sentiment(articles)
    
    # Generate comparative analysis
    comparative_analysis = generate_comparative_analysis(articles_with_sentiment)
    
    # Generate final sentiment analysis (based on the most frequent sentiment)
    final_sentiment_analysis = (
        f"{company_name}'s latest news coverage is mostly "
        f"{max(comparative_analysis['Sentiment Distribution'], key=comparative_analysis['Sentiment Distribution'].get)}."
    )
    
    # Generate Hindi TTS audio using gTTS
    audio_url = text_to_speech(comparative_analysis, final_sentiment_analysis, company_name)
    
    # Prepare the final output JSON
    output = {
        "Company": company_name,
        "Articles": [
            {
                "Title": article["title"],
                "Summary": article["summary"],
                "Sentiment": article["sentiment"],
                "Topics": article["topics"]
            } for article in articles_with_sentiment
        ],
        "Comparative Sentiment Score": {
            "Sentiment Distribution": comparative_analysis["Sentiment Distribution"],
            "Coverage Differences": comparative_analysis["Coverage Differences"],
            "Topic Overlap": comparative_analysis["Topic Overlap"]
        },
        "Final Sentiment Analysis": final_sentiment_analysis,
        "Audio": audio_url  # URL to the generated audio file
    }
    
    # Pretty-print the JSON output
    pretty_output = json.dumps(output, indent=4, ensure_ascii=False)
    return Response(content=pretty_output, media_type="application/json")
