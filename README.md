# AI Resume Chatbot (RAG + LangChain + FAISS)

An AI-powered chatbot that allows users to interact with their resumes using natural language. The system uses Retrieval-Augmented Generation (RAG) to analyze resume content and answer questions such as skills, projects, and experience.

---

## Features

- Upload resume in PDF format
- Ask questions about your resume
- Extract skills and experience automatically
- Resume summarization
- Semantic search using embeddings
- Retrieval-Augmented Generation (RAG)

---

## Tech Stack

Python  
LangChain  
FAISS Vector Database  
HuggingFace Transformers  
Sentence Transformers  
Streamlit  

---

## Architecture

Resume PDF
↓
PyPDFLoader
↓
Text Splitter
↓
Sentence Transformer Embeddings
↓
FAISS Vector Database
↓
Retriever
↓
FLAN-T5 Language Model
↓
Streamlit Chat Interface

---

## Installation

Clone the repository
