import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center;color:#4CAF50;'>Sentiment Analysis Web App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>AI-powered Sentiment Detection using NLP</p>",
    unsafe_allow_html=True
)

user_input = st.text_area("Enter your text:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
