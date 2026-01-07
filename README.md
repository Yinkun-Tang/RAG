# Game Series RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot focused on the Hitman video game series, combining semanticand lexical retrieval with section-aware reranking, providing answers with references to relevant source text chunks.

## Features

· Hybrid Retriever: Combines FAISS semantic search and BM25 lexical search using RRF (Reciprocal Rank Fusion)

· Section-Aware Reranking: Prioritizes key sections like Reception, Controversy, and Gameplay to improve answer relevance

· LLM Integration: Uses Google Gemini API to generate natural language responses based on retrieved documents

· Reference-Aware Answers: Responses include clear references to retrieved documents

· Streamlit Chat Interface: Interactive web interface for testing and querying the RAG system

## Installation

1. Clone the repository
2. Create a Python virtual environment and activate it
3. Install dependencies with ```requirements.txt```
4. Set your Gemini API key as an environment variable

· Linux/maxOS: ```export GEMINI_API_KEY="your_api_key"```

· Windows: ```set GEMINI_API_KEY=your_api_key```

## Usage

Launch Streamlit Chatbot using command ```streamlit run app.py``` under ```backend``` directory and ask any question related to Hitman game series.