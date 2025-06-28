
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV dataset
df = pd.read_csv("Chatbot Questions & Answers.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['clean_questions'] = df['Questions'].apply(clean_text)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df['clean_questions'])

def get_response(user_input, threshold=0.3):
    cleaned_input = clean_text(user_input)
    user_input_vector = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(user_input_vector, X)
    max_score = similarity.max()
    if max_score < threshold:
        return "I'm not sure about that. Can you please rephrase or ask something else?"
    index = similarity.argmax()
    return df.iloc[index]['Answers']

st.title("🎓 Kepler College Chatbot")
st.write("Ask anything about Kepler College and get answers based on available information.")

user_input = st.text_input("Type your question:")

if user_input:
    response = get_response(user_input)
    st.write("**Chatbot:**", response)
