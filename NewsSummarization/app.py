import streamlit as st
import requests

st.title("News Summarization & Sentiment Analysis")

# User Input
company_name = st.text_input("Enter Company Name", "Tesla")

if st.button("Fetch News"):
    with st.spinner("Fetching news..."):
        response = requests.post("https://your-api-url/fetch_news", json={"company": company_name})
        news_data = response.json()["articles"]

        if news_data:
            st.subheader("News Articles")
            for i, article in enumerate(news_data):
                st.write(f"**{i+1}. {article['title']}**")
                st.write(f"[Read More]({article['url']})")
                
            # Sentiment Analysis API Call
            sentiment_response = requests.post("https://your-api-url/analyze_sentiment", json={"articles": news_data})
            analyzed_data = sentiment_response.json()["articles"]

            st.subheader("Sentiment Analysis")
            for i, article in enumerate(analyzed_data):
                st.write(f"**{i+1}. {article['title']}** - *{article['sentiment']}*")

            # Convert summary to Hindi speech (API)
            st.subheader("Hindi Text-to-Speech")
            summary_text = " ".join([article['title'] for article in analyzed_data])
            tts_response = requests.post("https://your-api-url/tts", json={"text": summary_text})
            
            st.audio("output.mp3")

        else:
            st.error("No articles found. Try another company!")
