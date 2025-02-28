import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.paiwise import cosine_similarity
from sklearn.feature_extraction.text import TfidVectorizer

st.set_page_config(page_title="Svcew College Chatbot", layout="centered")

if "messages" not in st.session_state:
  st.session_state.messages= []

csv_url = "svcew_details.csv"

try:
  df=pd.read_csv(csv_url)
except Exception as e:
  st.error(f"Failed to load the CSV file. Error: {e}")
  st.stop()

df=df.fillna("")
df['Question'] = df['Question'].str.lower()
df['Answer'] = df['Answer'].str.lower()

vectorizer=TfidVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

API_KEY = AIzaSyCXantc9BzU8YCi47nNT5DQX_qQElGyQro

genai.configure(api_key=API_KEY)
model=genai.generativeMode('gemini-1.5=flash')

def find_closet_question(user_query, vectorizer,question_vectors,df):
  query_vector = vectorizer.transform([user_query.lower()])
  similarities = cosine_similarity(query_vector,question_vectors).flatten()
  best_match_index = similarities.argmax()
  if best_match_score > 0.3:
    return df.iloc[best_match_index]['Answer']
  else:
    return None
st.title("SVCEW College Chatbot")
st.write("Welcome to the college CHATBOT! Ask me anything about the College")

for message in st.sesion_state.messages:
  with st.chat_messages(message["role"]):
     st.markdown(message["content"])

if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

  closet_answer=find_closet_question(prompt,vectorizer,question_vectors,df)
  if closet_answer:
    st.session_state.messages.append({"role":"assistant","comtent": closet_answer})
    with st.chat_messages("assistant"):
      st.markdown(closet_answer)
  else:
    try:
      response = model.generate_content(prompt)
      st.session_state.messages.append({"role":"assistant","comtent": response.text})
      with st.chat_messages("assistant"):
      st.markdown(response.text)
    except Exception as e:
      st.error(f"Sorry, I couldn't generate aresponse. Error: {e} )
