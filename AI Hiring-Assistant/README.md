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

### 1. Clone the repository
```bash
git clone https://github.com/your-username/talent-scout-assistant.git
cd talent-scout-assistant
