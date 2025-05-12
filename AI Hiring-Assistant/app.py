import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

# System prompt template
SYSTEM_PROMPT = """
You are TalentScout, an AI hiring assistant for a recruitment agency specializing in technology placements. 
Your role is to conduct initial screening of candidates by gathering essential information and assessing their technical skills.

Follow these steps in every conversation:
1. Greet the candidate warmly and briefly explain your purpose.
2. Collect the following candidate details (ONE AT A TIME):
   - Full Name
   - Email Address
   - Phone Number
   - Years of Experience
   - Desired Position(s)
   - Current Location
   - Tech Stack (programming languages, frameworks, tools)
3. After collecting basic information, ask exactly 5 technical questions relevant to their tech stack and experience.
4. Maintain context throughout the conversation.
5. If the user says "goodbye", "exit", or similar, end the conversation gracefully.

STRICT RULES:
- Always ask ONE question at a time
- Never ask multiple questions in a single response
- Wait for the candidate to answer before asking the next question
- Ask exactly 5 technical questions after basic information is collected
- Be professional but friendly
- Don't proceed to technical questions until all basic information is collected
- For technical questions, make them relevant to the candidate's experience level
- Never prefix your responses with "Assistant:" or similar labels
"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm TalentScout, your AI hiring assistant. I'll help with your initial screening process. Could you please start by telling me your full name?"}]
if "collected_info" not in st.session_state:
    st.session_state.collected_info = {
        "full_name": None,
        "email": None,
        "phone": None,
        "experience": None,
        "position": None,
        "location": None,
        "tech_stack": None,
        "questions_asked": 0  # Track number of technical questions asked
    }
if "stage" not in st.session_state:
    st.session_state.stage = "greeting"  # greeting -> collecting_info -> technical_questions -> closing

# App title
st.title("TalentScout Hiring Assistant")

# Data privacy notice in sidebar
with st.sidebar:
    st.markdown("""
    **Data Privacy Notice**  
    The information you provide in this chat will be used solely for recruitment purposes. 
    We adhere to strict data protection regulations and will not share your details with 
    any third parties without your explicit consent.
    """)
    
    if st.button("Reset Conversation"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm TalentScout, your AI hiring assistant. I'll help with your initial screening process. Could you please start by telling me your full name?"}]
        st.session_state.collected_info = {
            "full_name": None,
            "email": None,
            "phone": None,
            "experience": None,
            "position": None,
            "location": None,
            "tech_stack": None,
            "questions_asked": 0
        }
        st.session_state.stage = "greeting"
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate AI response
def generate_response(prompt):
    # Combine system prompt with conversation history
    conversation_history = SYSTEM_PROMPT + "\n\nCurrent conversation:\n"
    for msg in st.session_state.messages:
        conversation_history += f"{msg['role']}: {msg['content']}\n"
    
    # Add the new user prompt
    conversation_history += f"user: {prompt}\n"
    
    # Generate response
    response = model.generate_content(conversation_history)
    
    # Remove any "Assistant:" prefixes from the response
    cleaned_response = response.text
    if cleaned_response.startswith("Assistant:"):
        cleaned_response = cleaned_response[len("Assistant:"):].strip()
    elif cleaned_response.startswith("Assistant"):
        cleaned_response = cleaned_response[len("Assistant"):].strip()
    
    return cleaned_response

# User input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check for exit commands
    if any(word in prompt.lower() for word in ["exit", "goodbye", "bye", "end", "stop"]):
        st.session_state.stage = "closing"
    
    # Check if all basic info is collected
    basic_info_collected = all(st.session_state.collected_info[key] is not None 
                           for key in ['full_name', 'email', 'phone', 'experience', 'position', 'location', 'tech_stack'])
    
    # Generate AI response
    with st.spinner("Thinking..."):
        response = generate_response(prompt)
        
        # If we're in technical questions stage, increment counter
        if st.session_state.stage == "technical_questions" and basic_info_collected:
            st.session_state.collected_info["questions_asked"] += 1
            
        # Move to technical questions stage if all basic info is collected
        if basic_info_collected and st.session_state.stage != "technical_questions":
            st.session_state.stage = "technical_questions"
            response = "Thank you for providing all the basic information. Now I'll ask you some technical questions to assess your skills."
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response)