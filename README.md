# AI Voice Diary 

Project Overview

AI Voice Diary is a voice-based chatbot that recognizes speech, detects emotions, and responds naturally. It also tracks emotion trends over time and provides an analysis of the user's emotional state.

Features:

✅ Voice Recognition: Uses Google Speech Recognition to capture user input.
✅ Emotion Detection: Analyzes emotions in text using a deep learning model.
✅ AI Chatbot: Generates responses based on user emotions using BlenderBot.
✅ Audio Response: Converts chatbot replies to speech using Google Text-to-Speech.
✅ Emotion Analysis Page: Plots trends and generates summaries.
✅ Profile Page: Displays user emotion trends.
✅ Real Chat Interface: Messages are displayed in chat bubbles.

# Installation & Setup

1. Install Dependencies

Ensure you have Python installed, then install the required packages:

pip install streamlit speechrecognition torch transformers gtts pygame matplotlib numpy

2. Run the Application

streamlit run app.py

3. Interact with the Chatbot

Click "Start Talking" to begin voice recording.
The chatbot will analyze emotions and respond accordingly.
View emotion trends in the Emotion Analysis tab.
Check user details in the Profile tab.

# Project Structure

Chatbot Interface: Voice input & chatbot responses.
Emotion Analysis: Graphs emotion trends.
Profile Page: Displays user insights.
No deployment is needed. The app runs locally via Streamlit. 🚀