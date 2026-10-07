# 🤖 Company Knowledge Assistant

A Hybrid RAG (Retrieval-Augmented Generation) chatbot built with Python, Streamlit, FAISS, HuggingFace Embeddings, and Ollama.

The system can answer questions using a company document knowledge base and can also respond to general questions using an LLM.

## ✨ Features

- PDF-based knowledge base
- Semantic search using FAISS
- HuggingFace embeddings
- Local LLM using Ollama
- Streamlit chatbot interface
- Displays document source and page number
- Supports general AI questions
- Chat history support

## 🧠 How It Works

```text
Company PDF Documents
        ↓
Text Chunking
        ↓
HuggingFace Embeddings
        ↓
FAISS Vector Database
        ↓
User Question
        ↓
Relevant Document Retrieval
        ↓
LLM (Ollama)
        ↓
Answer + Source

project Structure

rag-project/
│
├── data/
│   └── document.pdf
│
├── faiss_db/
│   ├── index.faiss
│   └── index.pkl
│
├── ingest.py
├── streamlit_app.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md




🛠️ Technologies Used
- Python
- LangChain
- Streamlit
- FAISS
- HuggingFace Sentence Transformers
- Ollama
- Llama 3.2



🚀 Installation
Clone the repository:git clone https://github.com/ishadi2002/rag-knowledge-assistant.git



Go to the project folder:
cd rag-knowledge-assistant

Create a virtual environment:
python -m venv venv

Activate it on Windows:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

🦙 Install Ollama
Install Ollama and download the model:
ollama pull llama3.2:3b

📚 Build the Knowledge Base
Add PDF documents to the data folder.
Then run:
python ingest.py

This creates the FAISS vector database.
▶️ Run the Application
streamlit run streamlit_app.py

Open the local Streamlit URL in your browser.
💬 Example
Question:
What are the working hours?

Answer:
The normal working hours are from 9:00 AM to 5:00 PM, Monday to Friday.
Source:
document.pdf — Page 1

The assistant can also answer general questions that are not available in the company knowledge base.

🔄 Updating the Knowledge Base
To add a new document:
1. Add the PDF file to the data folder.
2. Run:
python ingest.py
3. Restart the Streamlit application.


👩‍💻 Project Purpose
This project was developed as an internship task to demonstrate a basic Hybrid RAG system using document retrieval and a Large Language Model.

