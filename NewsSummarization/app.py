import streamlit as st
import requests
import os
import base64

API_URL = "http://127.0.0.1:8000/get_news/"

st.title("News Summarization & Sentiment Analysis")

company_name = st.text_input("Enter Company Name")

if st.button("Get News"):
    if company_name:
        with st.spinner("Fetching news..."):
            response = requests.post(API_URL, json={"company": company_name})
            data = response.json()

        if "error" in data:
            st.error(data["error"])
        else:
            st.subheader(f"News Articles for {data['company']}")
            for article in data["articles"]:
                st.markdown(f"**{article['title']}**")
                st.markdown(f"*{article['summary']}*")
                st.markdown(f"**Sentiment:** {article['sentiment']}")
                st.markdown("---")

            st.subheader("Sentiment Distribution")
            st.json(data["sentiment_summary"])

            st.subheader("Hindi Text-to-Speech Summary")
            audio_file = data["audio"]
            with open(audio_file, "rb") as file:
                audio_bytes = file.read()
                st.audio(audio_bytes, format="audio/mp3")

    else:
        st.warning("Please enter a company name")
