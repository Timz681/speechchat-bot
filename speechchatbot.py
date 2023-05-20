import nltk
import streamlit as st
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# Load and preprocess the dataset
with open("dataset.txt", 'r', encoding='utf8', errors='ignore') as file:
    lines = file.readlines()

questions = []
answers = []
for line in lines:
    if line.startswith('Q:'):
        question = line.replace('Q:', '').strip()
        questions.append(question)
    elif line.startswith('A:'):
        answer = line.replace('A:', '').strip()
        answers.append(answer)

lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(sentence.lower())
    words = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return words

corpus = [" ".join(preprocess(sentence)) for sentence in questions]

# Vectorize corpus
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Define chatbot function
def chatbot_response(user_input):
    user_input = " ".join(preprocess(user_input))
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, X)
    idx = similarities.argmax()
    return answers[idx]

# Define speech recognition function
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.write(f"Transcribed Speech: {text}")
        response = chatbot_response(text)
        st.write(f"Bot: {response}")
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand your speech.")
    except sr.RequestError:
        st.write("Sorry, I am having trouble accessing the speech recognition service.")

# Streamlit app
def main():
    # Initialize chat history
    chat_history = []

    st.title("Victoria Speech-Enabled Chatbot")
    # User input options
    option = st.sidebar.radio("Input Method:", ("Text", "Speech"))

    # Function to display chat history on sidebar
    def display_chat_history(chat_history):
        for message in chat_history:
            bubble_class = "user-bubble" if message["sender"] == "User" else "bot-bubble"
            timestamp = message["timestamp"]
            text = message["text"]
            st.sidebar.write(f'<div class="chat-bubble {bubble_class}"><small>{timestamp}</small><br>{text}</div>', unsafe_allow_html=True)

    if option == "Text":
        user_input = st.text_input("User:")
        if st.button("Send"):
            if user_input:
                # Add user input to chat history
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                chat_history.append({"sender": "User", "timestamp": timestamp, "text": user_input})

                response = chatbot_response(user_input)

                # Add bot response to chat history
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                chat_history.append({"sender": "Bot", "timestamp": timestamp, "text": response})

    elif option == "Speech":
        if st.button("Click here to speak"):
            user_input = transcribe_speech()
            if user_input:
                # Add user input to chat history
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                chat_history.append({"sender": "User", "timestamp": timestamp, "text": user_input})

                response = chatbot_response(user_input)

                # Add bot response to chat history
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                chat_history.append({"sender": "Bot", "timestamp": timestamp, "text": response})

    # Display chat history on sidebar
    st.sidebar.markdown('<div class="chat-history">', unsafe_allow_html=True)
    display_chat_history(chat_history)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Display chat history in main container
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    display_chat_history(chat_history)
    st.markdown('</div>', unsafe_allow_html=True)

# CSS styles for the chat interface
CSS = """
<style>
.chat-container {
    max-width: 800px;
    margin: 20px auto;
    padding: 20px;
    border-radius: 10px;
    background-color: #f7f7f7;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.chat-bubble {
    padding: 10px 15px;
    margin-bottom: 10px;
    border-radius: 10px;
    font-size: 16px;
    line-height: 1.4;
}

.user-bubble {
    background-color: #f2f2f2;
    color: #000;
    align-self: flex-start;
}

.bot-bubble {
    background-color: #007bff;
    color: #fff;
    align-self: flex-end;
}

.chat-history {
    max-height: 400px;
    overflow-y: auto;
}

.chat-input {
    width: 100%;
    padding: 10px;
    border: none;
    border-radius: 5px;
    background-color: #fff;
    box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
}

.chat-button {
    margin-top: 10px;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    background-color: #007bff;
    color: #fff;
    cursor: pointer;
}
</style>
"""

# Add CSS styles to the page
st.sidebar.markdown(CSS, unsafe_allow_html=True)
st.markdown(CSS, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
