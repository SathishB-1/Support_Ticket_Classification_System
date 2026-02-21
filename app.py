import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained model
model = joblib.load("ticket_model.pkl(1)")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# Streamlit UI
st.title("📩 Support Ticket Classification System")

st.write("Enter a support ticket description below:")

user_input = st.text_area("Ticket Description")

if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        prediction = model.predict([cleaned])
        st.success(f"Predicted Category: {prediction[0]}")