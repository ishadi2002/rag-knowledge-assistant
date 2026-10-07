import os
import glob
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_FOLDER = "data"
DB_FOLDER = "faiss_db"


# ---------------------------------
# Find all PDF files
# ---------------------------------
pdf_files = glob.glob(
    os.path.join(DATA_FOLDER, "*.pdf")
)

if not pdf_files:
    print("No PDF files found in the data folder.")
    exit()


print(f"Found {len(pdf_files)} PDF file(s).")


# ---------------------------------
# Load all PDFs
# ---------------------------------
documents = []

for pdf_file in pdf_files:

    print(
        f"Loading: {os.path.basename(pdf_file)}"
    )

    loader = PyPDFLoader(pdf_file)

    pages = loader.load()

    # Save clean filename in metadata
    for page in pages:

        page.metadata["file_name"] = (
            os.path.basename(pdf_file)
        )

    documents.extend(pages)


print(
    f"Total pages loaded: {len(documents)}"
)


# ---------------------------------
# Split documents into chunks
# ---------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(
    documents
)

print(
    f"Total chunks created: {len(chunks)}"
)


# ---------------------------------
# Embedding model
# ---------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------
# Create FAISS vector database
# ---------------------------------

# Remove old database before rebuilding
if os.path.exists(DB_FOLDER):
    shutil.rmtree(DB_FOLDER)


print("Creating vector database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local(
    DB_FOLDER
)


print("\n✅ Knowledge base created successfully!")
print(f"Saved to: {DB_FOLDER}")