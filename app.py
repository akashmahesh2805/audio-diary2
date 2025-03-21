import os
import streamlit as st
import speech_recognition as sr
from transformers import pipeline, BlenderbotTokenizer, BlenderbotForConditionalGeneration
from gtts import gTTS
import torch
import tempfile
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter

# Load emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=0 if torch.cuda.is_available() else -1)

# Load Facebook's open-source chatbot model
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Initialize session states
if "emotion_history" not in st.session_state:
    st.session_state["emotion_history"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [("bot", "Hello! How was your day?")]

# Function to generate chatbot response
def generate_response(user_input, emotion):
    """Generate chatbot response using BlenderBot."""
    prompt = f"User: {user_input}\nEmotion: {emotion}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Function to generate emotion trend summary
def generate_summary(emotion_history):
    """Generate a meaningful summary based on emotion trends."""
    if len(emotion_history) < 5:
        return "Not enough data to generate a meaningful summary yet."

    last_10_days = [entry["emotion"] for entry in emotion_history[-10:]]
    emotion_count = Counter(last_10_days)
    most_common = emotion_count.most_common(1)[0][0]

    if most_common == "joy":
        return "You've been feeling happier recently! Keep up the positive energy!"
    elif most_common == "sadness":
        return "It seems like you've been feeling a bit down. Hope things get better soon!"
    elif most_common == "anger":
        return "You've expressed some frustration recently. Maybe taking a break would help."
    elif most_common == "fear":
        return "There has been some anxiety in recent days. Consider practicing relaxation techniques."
    else:
        return "Your emotions have been quite mixed. Stay mindful and take care of yourself!"

# Function to recognize speech
recognizer = sr.Recognizer()
def recognize_speech():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand."
    except sr.RequestError:
        return "Speech recognition service unavailable."

# Process text input
def process_audio(text):
    emotion_result = emotion_classifier(text)[0]
    emotion = emotion_result["label"]
    bot_response = generate_response(text, emotion)
    return emotion, bot_response

# Function to play bot response
def speak(text):
    pygame.mixer.init()
    tts = gTTS(text)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    temp_audio.close()
    sound = pygame.mixer.Sound(temp_audio.name)
    sound.play()
    while pygame.mixer.get_busy():
        time.sleep(0.1)
    os.remove(temp_audio.name)

# Save emotions
def save_emotion_history(emotion, text):
    st.session_state["emotion_history"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d"),
        "emotion": emotion,
        "text": text
    })

# Streamlit UI
st.set_page_config(page_title="AI Voice Diary", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Emotion Analysis", "Profile"])

# Emotion Analysis Page
if page == "Emotion Analysis":
    st.title("Emotion Analysis")

    if not st.session_state["emotion_history"]:
        st.write("No emotion data available yet. Start a conversation!")
    else:
        timestamps = [entry["timestamp"] for entry in st.session_state["emotion_history"]]
        emotions = [entry["emotion"] for entry in st.session_state["emotion_history"]]

        # Count emotions per day
        unique_days = sorted(set(timestamps))
        emotions_list = ["joy", "sadness", "surprise", "anger", "fear"]
        emotion_data = {e: [0] * len(unique_days) for e in emotions_list}
        
        for i, day in enumerate(unique_days):
            for entry in st.session_state["emotion_history"]:
                if entry["timestamp"] == day:
                    emotion_data[entry["emotion"]][i] += 1

        # Plot actual emotion data
        plt.figure(figsize=(8, 5))
        for e in emotions_list:
            plt.plot(unique_days, emotion_data[e], label=e, marker="o")
        plt.xlabel("Date")
        plt.ylabel("Emotion Frequency")
        plt.title("Emotion Trends Over Time")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

        # Generate AI-based summary
        summary = generate_summary(st.session_state["emotion_history"])
        st.write(f"**Summary:** {summary}")

# Profile Page
elif page == "Profile":
    st.title("User Profile")
    st.image("profile_icon.png", width=100)
    st.write("**Name:** Demo User")

    if st.session_state["emotion_history"]:
        emotion_count = Counter([entry["emotion"] for entry in st.session_state["emotion_history"]])
        most_common_emotion = max(emotion_count, key=emotion_count.get)
        st.write(f"**Emotion Trends:** You have been mostly expressing `{most_common_emotion}`.")
    else:
        st.write("**Emotion Trends:** No data yet.")

# Chatbot Page
elif page == "Chatbot":
    st.title("AI Voice Diary")

    # Chat Messages Section with Bubble Styling
    chat_container = st.container()
    with chat_container:
        st.write("### Chat History:")
        chat_html = """
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                gap: 10px;
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
            }
            .message {
                padding: 12px;
                border-radius: 18px;
                max-width: 70%;
                word-wrap: break-word;
                font-size: 16px;
            }
            .bot {
                align-self: flex-start;
                background-color: #f0f0f0;
                color: black;
            }
            .user {
                align-self: flex-end;
                background-color: #008080;
                color: white;
            }
        </style>
        <div class='chat-container'>
        """

        for role, content in st.session_state["messages"]:
            message_class = "user" if role == "user" else "bot"
            chat_html += f"<div class='message {message_class}'>{content}</div>"

        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # Fixed Start/Stop Buttons at the Bottom
    st.markdown("---")
    button_col1, button_col2, button_col3 = st.columns([2, 1, 2])
    with button_col2:
        start_button = st.button("🎤 Start Talking")
        stop_button = st.button("🛑 Stop Talking")

    # Speech Processing
    if start_button:
        user_text = recognize_speech()
        if user_text and user_text not in ["Could not understand.", "Speech recognition service unavailable."]:
            st.session_state["messages"].append(("user", user_text))
            emotion, bot_response = process_audio(user_text)
            st.session_state["messages"].append(("bot", f"{bot_response} (Emotion: {emotion})"))
            save_emotion_history(emotion, user_text)
            speak(bot_response)
            st.rerun()
