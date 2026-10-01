# RAG PDF Chatbot

## Project Description

This project is a simple RAG (Retrieval-Augmented Generation) chatbot built with Python and Streamlit.

The user can upload a PDF document and ask questions about its content. The chatbot retrieves relevant information from the uploaded PDF and uses Gemini AI to generate an answer.

## Features

* Upload a PDF document
* Extract text from the PDF
* Split the document into chunks
* Create text embeddings
* Search relevant information using FAISS
* Generate answers using Gemini AI
* Simple Streamlit interface

## Technologies Used

* Python
* Streamlit
* PyPDF
* Sentence Transformers
* FAISS
* Google Gemini API

## How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file and add your Gemini API key:

```text
GOOGLE_API_KEY=your_api_key_here
```

3. Run the application:

```bash
streamlit run app.py
```

4. Upload a PDF and ask questions about the document.

## Important

Do not upload the `.env` file to GitHub because it contains the API key.

## Project Purpose

The purpose of this project is to demonstrate how RAG can be used to create a document-based question-answering chatbot. 