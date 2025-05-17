# ResoluteAI.assesment.


# RAG-Based Document Chat Application

## Overview
This application allows users to upload multi document PDF, DOCX, and TXT documents, process their content, and interact with them via a chatbot interface powered by **Google Gemini AI** and **FAISS-based retrieval**.

## Features
- Extracts text from PDF, DOCX, and TXT files.
- Segments text into meaningful chunks.
- Builds a **FAISS** vector store for efficient text retrieval.
- Provides an interactive chat interface for querying document content.
- Uses **TF-IDF vectorization** for intelligent context retrieval.
- Generates AI-powered responses using **Google Gemini API**.

## Dependencies
Install the required libraries using:

```bash
pip install streamlit google-generativeai pymupdf python-docx numpy scikit-learn faiss-cpu



- Configure the Google Gemini API Key:
- Open app.py and replace "YOUR_GEMINI_API_KEY" with your actual API key


Usage
Run the Streamlit app:
streamlit run app.py


Steps:
- Upload PDF, DOCX, or TXT files.
- The documents will be processed and indexed automatically.
- Enter a question related to the document content.
- View AI-generated answers and chat history.


├── app.py               # Main Streamlit application file
├── README.md            # Project documentation
├── requirements.txt     # List of required Python packages
└── sample_docs/         # Example files for testing
