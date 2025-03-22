import requests
import os
from textblob import TextBlob
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from gtts import gTTS
# Download NLTK data files
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")


# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
API_KEY = os.getenv("NEWSAPI_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Please set the NEWSAPI_API_KEY environment variable.")

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Fetch news articles using NewsAPI
def fetch_news(company_name):
    API_ENDPOINT = "https://newsapi.org/v2/everything"
    
    params = {
        "q": company_name,
        "apiKey": API_KEY,
        "pageSize": 10,  # Fetch 10 articles
        "language": "en"
    }
    
    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [{
            "title": article["title"],
            "summary": article["description"],
            "content": article["content"]
        } for article in articles]
    else:
        print(f"Error: {response.status_code} - {response.text}")  # Debug statement
        return []

# Perform sentiment analysis using TextBlob
def analyze_sentiment(articles):
    for article in articles:
        blob = TextBlob(article["content"])
        sentiment_score = blob.sentiment.polarity
        if sentiment_score > 0:
            article["sentiment"] = "Positive"
        elif sentiment_score < 0:
            article["sentiment"] = "Negative"
        else:
            article["sentiment"] = "Neutral"
        
        # Extract topics dynamically
        article["topics"] = extract_topics(article["content"])
    return articles

# Extract topics from content
def extract_topics(content):
    if not content:
        return []
    
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(content.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Count word frequency and extract top 3 topics
    word_freq = Counter(filtered_words)
    topics = [word for word, _ in word_freq.most_common(3)]
    return topics

# Generate comparative analysis with relaxed common topics
def generate_comparative_analysis(articles):
    sentiment_distribution = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }
    
    for article in articles:
        sentiment_distribution[article["sentiment"]] += 1
    
    # Extract all topics and count their occurrences across articles
    topic_counter = defaultdict(int)
    for article in articles:
        for topic in article["topics"]:
            topic_counter[topic] += 1
    
    # Common topics are those that appear in at least 2 articles
    common_topics = [topic for topic, count in topic_counter.items() if count >= 2]
    
    # Calculate unique topics for each article
    unique_topics = {}
    for i, article in enumerate(articles):
        # Get topics that are unique to this article
        unique_topics[f"Unique Topics in Article {i+1}"] = [
            topic for topic in article["topics"] if topic_counter[topic] == 1
        ]
    
    # Dynamically generate coverage differences
    coverage_differences = []
    for i in range(len(articles) - 1):
        article1 = articles[i]
        article2 = articles[i + 1]
        
        # Generate comparison based on topics and sentiment
        comparison = (
            f"Article {i+1} discusses {', '.join(article1['topics'])} with a {article1['sentiment'].lower()} sentiment, "
            f"while Article {i+2} discusses {', '.join(article2['topics'])} with a {article2['sentiment'].lower()} sentiment."
        )
        
        # Generate impact based on sentiment
        if article1["sentiment"] == "Positive" and article2["sentiment"] == "Negative":
            impact = (
                "The first article highlights positive developments, boosting confidence in the company's growth, "
                "while the second article raises concerns about potential challenges."
            )
        elif article1["sentiment"] == "Negative" and article2["sentiment"] == "Positive":
            impact = (
                "The first article raises concerns about challenges, while the second article highlights positive developments, "
                "boosting confidence in the company's growth."
            )
        else:
            impact = "Both articles provide a balanced perspective on the company's current situation."
        
        coverage_differences.append({
            "Comparison": comparison,
            "Impact": impact
        })
    
    comparative_analysis = {
        "Sentiment Distribution": sentiment_distribution,
        "Coverage Differences": coverage_differences,
        "Topic Overlap": {
            "Common Topics": common_topics,
            **unique_topics  # Merge unique topics for each article
        }
    }
    
    return comparative_analysis

# Function to generate Hindi text-to-speech audio using gTTS and save it as an MP3 file
def text_to_speech(comparative_analysis, final_sentiment_analysis, company_name):
    # Define the MP3 filename
    mp3_filename = f"static/{company_name}_summary_audio.mp3"

    # Extract components for the summary text
    sentiment_distribution = comparative_analysis["Sentiment Distribution"]
    coverage_differences = comparative_analysis["Coverage Differences"]
    topic_overlap = comparative_analysis["Topic Overlap"]

    # Gather unique topics from keys that start with "Unique Topics"
    unique_topics_list = []
    for key, value in topic_overlap.items():
        if key.startswith("Unique Topics"):
            unique_topics_list.append(f"{key}: {', '.join(value)}")
    unique_topics_str = "; ".join(unique_topics_list) if unique_topics_list else "None"

    # Create a text summary
    summary_text = (
        f"Sentiment Distribution: {sentiment_distribution['Positive']} positive, "
        f"{sentiment_distribution['Negative']} negative, and {sentiment_distribution['Neutral']} neutral articles. "
        f"Coverage Differences: {coverage_differences[0]['Comparison']} {coverage_differences[0]['Impact']} "
        f"Common Topics: {', '.join(topic_overlap.get('Common Topics', []))}. "
        f"Unique Topics: {unique_topics_str}. "
        f"Final Sentiment Analysis: {final_sentiment_analysis}"
    )

    # Translate the summary text into Hindi
    translated_text = GoogleTranslator(source="auto", target="hi").translate(summary_text)

    # Generate speech from the translated text
    tts = gTTS(translated_text, lang="hi", slow=False)

    # Save the MP3 file
    tts.save(mp3_filename)

    return f"http://127.0.0.1:8000/static/{company_name}_summary_audio.mp3"
