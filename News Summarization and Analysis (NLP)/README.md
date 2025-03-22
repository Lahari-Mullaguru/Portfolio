# News Summarization & Sentiment Analysis App

## Features:
Fetches news articles related to a company  
Performs sentiment analysis  
Provides comparative sentiment insights  
Converts summary into **Hindi speech**  
User-friendly **Streamlit UI**  

---

## **Setup Instructions**
**Clone the Repository**
```bash
git clone https://github.com/Lahari-Mullaguru/Portfolio.git
cd Portfolio/NewsSummary

** Install Dependencies**
pip install -r requirements.txt

**Run Flask API Server**
python api.py

** Run Streamlit App**
streamlit run app.py

=======
# News Summarization and Sentiment Analysis Application

This application extracts news articles related to a given company, performs sentiment analysis, conducts a comparative analysis, and generates a Hindi text-to-speech summary.

## Features

- **News Extraction**: Fetches news articles using the NewsAPI.
- **Sentiment Analysis**: Analyzes sentiment (positive, negative, neutral) using TextBlob.
- **Comparative Analysis**: Compares sentiment and topics across multiple articles.
- **Text-to-Speech**: Converts the summary into Hindi speech using gTTS.
- **Web Interface**: Provides a simple web-based interface using Streamlit.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lahari-Mullaguru/DataScience.git
   cd DataScience/NewsSummarization
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Set up environment variables**:
   Create a .env file in the root directory.
   I already have the .env file in my repository with my API key, you can test it further using any other key.
   Add your NewsAPI key:
   ```bash
   NEWSAPI_API_KEY=your_api_key_here
5. **Run the FastAPI backend**:
   ```bash
   python -m uvicorn api:app --reload
6. **Request via curl command**:
   Open the other terminal and enter the curl command
   ```bash
   curl -X GET "http://127.0.0.1:8000/analyze-news?company_name=Tesla"
7. **Request via Postman**:
   - Create a new request, Click on New HTTP Request.
   - Set up the request: Request Method: Select GET.
   ```bash
   http://127.0.0.1:8000/analyze-news?company_name=Tesla
8. **Run the Streamlit frontend**:
   ```bash
   python -m streamlit run app.py
9. **Access the application**:
   Open your browser and go to http://localhost:8501.
   Enter a company name and click "Analyze News"
>>>>>>> b965305cbfd233045619b5dac7e87fe8f873b497:NewsSummary/README.md
