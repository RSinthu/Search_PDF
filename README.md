# PDF Search Engine

A PDF search application that uses embeddings and a language model to answer questions about the content of your PDF documents.

## Overview

This project provides a Streamlit web interface that allows users to ask questions about PDF documents uploaded to the system. It uses:

- LangChain for document processing and retrieval
- Hugging Face embeddings (all-MiniLM-L6-v2) for vectorization
- LaMini-T5-738M model for generating answers
- ChromaDB for vector storage
- Streamlit for the web interface

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install langchain langchain-community langchain-huggingface streamlit transformers torch chromadb accelerate
   ```
3. Download the LaMini-T5-738M model from Hugging Face
4. Create a `docs` folder and add your PDF files

## Usage

1. First, ingest your PDF documents:
   ```
   python ingest.py
   ```

2. Run the web application:
   ```
   streamlit run app.py
   ```

3. Enter your questions in the text area and click "Get Answer"

## Project Structure

- `ingest.py`: Processes PDF files and creates vector embeddings
- `app.py`: Streamlit web application for asking questions
- `db/`: Directory where ChromaDB stores vector embeddings
- `docs/`: Directory where PDF files should be placed

## Note

Make sure you have enough system resources to run the language model. The application uses model offloading to manage memory usage.
