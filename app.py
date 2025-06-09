import streamlit as st
import pickle
import re
import string
import requests
import io

model_url = 'https://github.com/parvathykrishna05/fake-news-detection-ml/releases/download/v1.o/model.1.pkl'
tfidf_url = 'https://github.com/parvathykrishna05/fake-news-detection-ml/releases/download/v1.o/tfidf.1.pkl'

# Function to load pickle files from URL
def load_pickle_from_url(url, file_description):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return pickle.load(io.BytesIO(response.content))
        else:
            st.error(f"Failed to download {file_description}. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error loading {file_description}: {str(e)}")
        return None

# Load model and vectorizer
model = load_pickle_from_url(model_url, "model file")
tfidf = load_pickle_from_url(tfidf_url, "TF-IDF vectorizer")

# Stop if loading failed
if model is None or tfidf is None:
    st.stop()

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
