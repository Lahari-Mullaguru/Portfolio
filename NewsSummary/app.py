import streamlit as st
import requests

# Set the title of the app
st.title("News Summarization and Sentiment Analysis")

# Input for company name
company_name = st.text_input("Enter the company name (e.g., Tesla):")

# Button to trigger the analysis
if st.button("Analyze News"):
    if company_name:
        # Call the FastAPI backend
        response = requests.get(f"http://127.0.0.1:8000/analyze-news?company_name={company_name}")
        
        if response.status_code == 200:
            result = response.json()
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display the results
                st.subheader(f"Analysis for {company_name}")

                # Display articles inside dropdowns
                st.write("### Articles")
                for i, article in enumerate(result["Articles"], 1):
                    with st.expander(f"Article {i}: {article['Title']}"):
                        st.write(f"**Summary:** {article['Summary']}")
                        st.write(f"**Sentiment:** {article['Sentiment']}")
                        st.write(f"**Topics:** {', '.join(article['Topics'])}")

                # Display Sentiment Distribution
                st.write("### Comparative Sentiment Score")
                st.write("### Sentiment Distribution")
                sentiment_distribution = result["Comparative Sentiment Score"]["Sentiment Distribution"]
                st.write(f"Positive: {sentiment_distribution['Positive']}")
                st.write(f"Negative: {sentiment_distribution['Negative']}")
                st.write(f"Neutral: {sentiment_distribution['Neutral']}")

                # Display Coverage Differences
                st.write("### Coverage Differences")
                for coverage in result["Comparative Sentiment Score"]["Coverage Differences"]:
                    st.write(f"{coverage['Comparison']}")
                    st.write(f"{coverage['Impact']}")
                    st.write("---")

                # Display Topic Overlap
                st.write("### Topic Overlap")
                topic_overlap = result["Comparative Sentiment Score"]["Topic Overlap"]

                # Display Common Topics
                common_topics = ", ".join(topic_overlap.get("Common Topics", []))
                st.write(f"**Common Topics:** {common_topics if common_topics else 'None'}")

                # Display Unique Topics
                for key, value in topic_overlap.items():
                    if key.startswith("Unique Topics"):
                        st.write(f"**{key}:** {', '.join(value)}")

                # Display final sentiment analysis
                st.subheader("Final Sentiment Analysis")
                st.write(result["Final Sentiment Analysis"])

                # Display audio player
                st.subheader("Hindi Text-to-Speech Summary")
                st.audio(result["Audio"])
        else:
            st.error("Failed to fetch news. Please try again.")
    else:
        st.warning("Please enter a company name.")
