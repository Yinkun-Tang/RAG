import streamlit as st
from google import genai
from hybrid_retriever import HybridRetriever
import os

st.set_page_config(page_title="Hitman RAG Chatbot")
st.title("Hitman Series RAG Chatbot")

client = genai.Client()  # API key taken from GEMINI_API_KEY environment variable

retriever = HybridRetriever(
    faiss_index_path="../data/processed/hitman_faiss.index",
    metadata_path="../data/processed/hitman_index_mapping.json",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Hitman games:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    docs = retriever.search(prompt, final_top_k=5)
    context_text = ""
    references = []
    for i, doc in enumerate(docs, start=1):
        context_text += f"[{i}] {doc['text']}\n"
        references.append(f"{i}. {doc['page_title']} [{doc['section']}] - {doc['url']}")

    system_prompt = (
        "You are an expert on the Hitman video game series. "
        "Answer concisely using the provided context. "
        "Include inline citations using [1], [2], etc. referencing the context."
    )
    full_prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {prompt}"

    with st.chat_message("assistant"):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        answer_text = response.text
        st.markdown(answer_text)

    with st.expander("References"):
        for ref in references:
            st.markdown(ref)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
