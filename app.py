
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# Gemini API Key
API_KEY = "AQ.Ab8RN6LJPCgceNihwYtyp-qD001MKht7S5uqtB2a5TbL28f-Zg"

client = genai.Client(api_key=API_KEY)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# App title
st.title("My RAG Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    # Read PDF
    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        text += page.extract_text() or ""

    # Create chunks
    chunks = []

    for i in range(0, len(text), 500):
        chunks.append(text[i:i + 500])

    # Create embeddings
    embeddings = model.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # Ask question
    question = st.text_input("Ask a question:")

    if st.button("Get Answer"):

        if not question:
            st.warning("Please enter a question.")

        else:

            # Convert question into embedding
            question_embedding = model.encode([question])

            # Search similar chunks
            distances, indices = index.search(question_embedding, 3)

            # Create context
            context = ""

            for i in indices[0]:
                context += chunks[i] + "\n"

            # Create prompt
            prompt = f"""
Answer the question using only the context below.
If the answer is not present in the context, say:
"I could not find the answer in the uploaded document."

Context:
{context}

Question:
{question}
"""

            # Send to Gemini
            try:

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                st.write(response.text)

            except Exception as e:

                st.error("Gemini could not generate an answer.")
                st.write(str(e))
