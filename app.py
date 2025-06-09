import streamlit as st
import pickle
import re
import string
import os

# Load model and vectorizer from local files
try:
    with open('model.1.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('tfidf.1.pkl', 'rb') as file:
        tfidf = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"File not found: {str(e)}")
    st.stop()

# Rest of the code remains the same
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

st.title("📰 Fake News Detection")
st.subheader("Enter news content to check its authenticity.")

input_text = st.text_area("News Text")

if st.button("Check"):
    if input_text:
        cleaned = clean_text(input_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("🚫 Fake News")
    else:
        st.warning("Please enter some text.")
