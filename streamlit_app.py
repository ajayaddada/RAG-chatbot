import streamlit as st
from chatbot import RAGChatbot
import tempfile
import os

st.markdown(
    "<h1 style='text-align: center;'>Universal AI</h1>",
    unsafe_allow_html=True
)

bot = RAGChatbot()

col1, col2 = st.columns([6, 1])
with col1:
    q = st.text_input(
        "",
        placeholder="How can I help you?"
    )
with col2:
    uploaded = st.file_uploader(
        "",
        type=["txt", "pdf", "csv", "docx", "jpg", "jpeg", "png", "gif"],
        label_visibility="collapsed"
    )

docs = False
if uploaded:
    suffix = '.' + uploaded.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        docs = [tmp.name]
    bot.load_files(docs)
    st.success(f"File '{uploaded.name}' processed!", icon="✅")

if q:
    if uploaded and docs:
        answer = bot.ask(q)
    else:
        answer = bot.ask_with_web(q)
    st.markdown(f"**Answer:** {answer}")

# Make the uploader visually compact
st.markdown("""
<style>
div[data-testid="stFileUploader"] > label {display: none;}
</style>
""", unsafe_allow_html=True)
