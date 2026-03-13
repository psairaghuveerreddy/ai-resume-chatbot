import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------
# Load Local LLM
# -------------------------

generator = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",
    max_new_tokens=200,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=generator)

# -------------------------
# Custom Prompt
# -------------------------

template = """
You are an AI assistant that analyzes resumes.

Use the resume information below to answer the question clearly.

Resume:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# -------------------------
# Streamlit UI
# -------------------------

st.title("AI Resume Chatbot (RAG + LangChain + FAISS)")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:

    with open("resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader("resume.pdf")
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Database
    db = FAISS.from_documents(docs, embeddings)

    # Retrieval QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    question = st.text_input("Ask something about your resume")

    if question:
        response = qa({"query": question})
        answer = response["result"]

        # Clean answer if prompt text appears
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        st.write(answer)