import streamlit as st
import pickle
import requests
import io
import re
import string

model_url = 'https://raw.githubusercontent.com/parvathykrishna05/fake-news-detection-ml/refs/heads/main/model%20(1).pkl'
tfidf_url = 'https://raw.githubusercontent.com/parvathykrishna05/fake-news-detection-ml/refs/heads/main/tfidf%20(1).pkl'

model = pickle.load(io.BytesIO(requests.get(model_url).content))
tfidf = pickle.load(io.BytesIO(requests.get(tfidf_url).content))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("📰 Fake News Detection")
st.subheader("Enter news content to check its authenticity.")

input_text = st.text_area("News Text")

if st.button("Check"):
    cleaned = clean_text(input_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        st.success("✅ Real News")
    else:
        st.error("🚫 Fake News")
