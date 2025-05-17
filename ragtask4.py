import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF for PDF reading
from docx import Document  # For DOCX reading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import os

# Configure Gemini API
genai.configure(api_key="api key")  # Replace with your actual API key

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')  # Use the appropriate model name

# Step 1: Extract text from different file types
def extract_text(file):
    file_ext = os.path.splitext(file.name)[1].lower()
    text = ""
    
    if file_ext == ".pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    
    elif file_ext == ".docx":
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    
    elif file_ext == ".txt":
        text = file.read().decode("utf-8")
    
    return text

# Step 2: Chunk text
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 3: Build FAISS vector index
def build_vector_store(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks).toarray()
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype('float32'))
    return index, vectorizer, chunks

# Step 4: Search top-k similar chunks
def retrieve_context(query, index, vectorizer, chunks, top_k=3):
    query_vec = vectorizer.transform([query]).toarray().astype('float32')
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

# Streamlit app
st.title("RAG-based Document Chat Application")
st.write("Upload your documents (PDF, DOCX, TXT) and ask questions about their content.")

# File uploader for multiple documents
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Initialize session state for storing index and chat history
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.vectorizer = None
    st.session_state.chunks = []
    st.session_state.chat_history = []

# Process uploaded files
if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        text = extract_text(file)
        all_text += text + "\n"
    
    # Chunk and index the combined text
    chunks = chunk_text(all_text)
    index, vectorizer, chunk_store = build_vector_store(chunks)
    
    # Store in session state
    st.session_state.vector_store = index
    st.session_state.vectorizer = vectorizer
    st.session_state.chunks = chunk_store
    st.success("Documents processed and indexed successfully!")

# Chat interface
st.subheader("Chat with your documents")
user_input = st.text_input("Ask a question about the documents:")

if user_input and st.session_state.vector_store:
    # Retrieve context
    context_chunks = retrieve_context(user_input, st.session_state.vector_store, 
                                   st.session_state.vectorizer, st.session_state.chunks)
    context = "\n".join(context_chunks)
    
    # Build prompt
    prompt = f"""You are answering based on the document below:

Context:
{context}

Question: {user_input}
Answer concisely and accurately based on the provided context.
"""
    
    try:
        # Send to Gemini
        response = model.generate_content(prompt)
        answer = response.text
        
        # Update chat history
        st.session_state.chat_history.append({"question": user_input, "answer": answer})
        
        # Display response
        st.write("**Answer:**")
        st.write(answer)
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")
        st.markdown("---")

# Instructions for running the app
st.sidebar.header("How to Use")
st.sidebar.write("1. Upload one or more PDF, DOCX, or TXT files.")
st.sidebar.write("2. Wait for the documents to be processed and indexed.")
st.sidebar.write("3. Enter your question in the text box.")
st.sidebar.write("4. View the response and chat history below.")
