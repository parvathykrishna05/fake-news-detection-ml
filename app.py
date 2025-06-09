import streamlit as st
import pickle
import re
import string

# Load model and TF-IDF vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf.pkl", "rb") as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Streamlit UI
st.title("📰 Fake News Detection App")
st.markdown("Enter a news headline or paragraph to check its authenticity using a machine learning model trained on real-world data.")

# Input area
input_text = st.text_area("🔍 Paste news content below:")

# Predict button
if st.button("Check"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(input_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ This looks like **Real News**.")
        else:
            st.error("🚫 This looks like **Fake News**.")
