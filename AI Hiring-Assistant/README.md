# TalentScout: AI Hiring Assistant

**TalentScout** is a Streamlit-based AI assistant designed to streamline the initial screening process of tech candidates for recruitment agencies. It uses Google's Gemini Pro (via the `google.generativeai` SDK) to conduct friendly, professional conversations, gather essential candidate information, and assess technical capabilities through tailored questions.

## Features

- AI-powered initial screening assistant for tech recruitment
- Collects key candidate details step-by-step
- Asks 5 personalized technical questions based on experience and tech stack
- Maintains context and professionalism throughout the interaction
- Built with Streamlit for a conversational UI experience
- Respects data privacy with a sidebar notice and reset functionality

## Tech Stack

- [Python 3.10+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Google Generative AI SDK](https://pypi.org/project/google-generativeai/)
- [dotenv](https://pypi.org/project/python-dotenv/)


## Setup Instructions
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
2. **Set up environment variables**:
   Create a .env file in the root directory.
   I already have the .env file in my repository with my API key.
   Add your API key:
   ```bash
   GOOGLE_API_KEY=your api key here
3. **Run the Streamlit application**:
   ```bash
   python -m streamlit run app.py
4. **Access the application**:
   Open your browser and go to http://localhost:8501.
   start the conversation.
