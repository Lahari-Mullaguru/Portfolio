import streamlit as st
import requests

st.title("News Summarization & Sentiment Analysis")

company_name = st.text_input("Enter Company Name", "Tesla")

if st.button("Fetch News"):
    response = requests.post("http://127.0.0.1:5000/fetch_news", json={"company": company_name})
    
    if response.status_code == 200:
        data = response.json()
        articles = data["articles"]
        sentiment_analysis = data["sentiment_analysis"]

        st.subheader("News Articles")
        for article in articles:
            st.write(f"**{article['title']}**")
            st.write(f"{article['summary']}")
            st.write(f"[Read More]({article['url']})")
            st.write(f"Sentiment: {article['sentiment']}")
            st.write("---")

        st.subheader("Comparative Sentiment Analysis")
        st.write(sentiment_analysis)

        # Convert Summary to Hindi TTS
        text_summary = " ".join([article["summary"] for article in articles])
        tts_response = requests.post("http://127.0.0.1:5000/tts", json={"text": text_summary})

        if tts_response.status_code == 200:
            st.subheader("Hindi Audio Summary")
            st.audio("static/output.mp3")
    else:
        st.error("Error fetching news.")
