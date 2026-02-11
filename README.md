# 📘 RAG Project using Ollama + LangChain + FAISS

This project demonstrates a simple implementation of Retrieval-Augmented Generation (RAG).  
It takes a user’s question, searches for the most relevant information using FAISS, and generates a final answer using the TinyLlama model running locally through Ollama.

---

##  How to Run the Project

1️. Install required Python packages
```bash
pip install langchain langchain-ollama langchain-community faiss-cpu

2️. Install Ollama

Download Ollama from:
https://ollama.com/download

3️. Pull required models
ollama pull tinyllama
ollama pull nomic-embed-text

4️. Run the script
python rag.py

 Technologies Used

LangChain

Ollama (TinyLlama)

FAISS

Nomic Embeddings

Example

User:

What is LangChain?


RAG Answer:

LangChain is a framework designed for building applications powered by large language models.
