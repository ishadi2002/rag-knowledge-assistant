import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# 1. Load PDF
loader = PyPDFLoader("data/document.pdf")
documents = loader.load()


# 2. Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create or load vector database
db_path = "faiss_db"

if os.path.exists(db_path):

    print("Loading existing vector database...")

    vectorstore = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    print("Creating new vector database...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(db_path)

    print("Vector database saved!")


# 5. Connect local LLM
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


print("\nRAG Assistant is ready!")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ").strip()

    command = question.lstrip(":").strip().lower()

    if command in ["exit", "quit"]:
        print("RAG Assistant stopped.")
        break

    if question == "":
        print("Please enter a question.\n")
        continue

    results = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
You are a question-answering assistant.

Use ONLY the context below to answer the question.

If the answer is present in the context, answer directly.
If the information is not in the context, say:
"I don't know based on the provided document."

Keep the answer short and clear.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nAssistant:", response.content)
    print()