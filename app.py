import streamlit as st
import pickle
import re
import string

model = pickle.load(open('model(1).pkl', 'rb'))
tfidf = pickle.load(open('tfidf(1).pkl', 'rb'))

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
    cleaned = clean_text(input_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        st.success("✅ Real News")
    else:
        st.error("🚫 Fake News")
