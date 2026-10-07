import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


DB_FOLDER = "faiss_db"


# ---------------------------------
# Page setup
# ---------------------------------
st.set_page_config(
    page_title="Company Knowledge Assistant",
    page_icon="🤖"
)

st.title("🤖 Company Knowledge Assistant")

st.write(
    "Ask questions about the company knowledge base "
    "or ask general questions."
)


# ---------------------------------
# Check knowledge base
# ---------------------------------
if not (
    os.path.exists(os.path.join(DB_FOLDER, "index.faiss"))
    and
    os.path.exists(os.path.join(DB_FOLDER, "index.pkl"))
):

    st.error(
        "Knowledge base not found. "
        "Please run: python ingest.py"
    )

    st.stop()


# ---------------------------------
# Load models + FAISS
# ---------------------------------
@st.cache_resource
def load_system():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        DB_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    return vectorstore, llm


vectorstore, llm = load_system()


# ---------------------------------
# Chat history
# ---------------------------------
if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ---------------------------------
# User question
# ---------------------------------
question = st.chat_input(
    "Ask anything..."
)


if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)


    # ---------------------------------
    # Retrieve relevant document chunks
    # ---------------------------------
    results = vectorstore.similarity_search(
        question,
        k=4
    )


    context_parts = []
    sources = []


    for doc in results:

        page_number = (
            doc.metadata.get("page", 0) + 1
        )

        file_name = doc.metadata.get(
            "file_name",
            "Document"
        )

        context_parts.append(
            f"""
File: {file_name}
Page: {page_number}

{doc.page_content}
"""
        )

        sources.append(
            (file_name, page_number)
        )


    context = "\n\n".join(
        context_parts
    )


    # Remove duplicate sources
    sources = list(
        dict.fromkeys(sources)
    )


    # ---------------------------------
    # Decide:
    # document answer or general AI
    # ---------------------------------
    router_prompt = f"""
You are a routing assistant.

Decide whether the user's question can be
answered clearly using the document context.

Document context:
{context}

User question:
{question}

Rules:

If the document clearly contains enough
information to answer the question,
respond only:

DOCUMENT

If the question is casual conversation
or requires general knowledge,
respond only:

GENERAL

If the document does not clearly contain
the answer,
respond only:

GENERAL

Do not explain your decision.

Answer only DOCUMENT or GENERAL.
"""


    router_response = llm.invoke(
        router_prompt
    )

    route = (
        router_response.content
        .strip()
        .upper()
    )


    # ---------------------------------
    # Document / RAG answer
    # ---------------------------------
    if route.startswith("DOCUMENT"):

        answer_prompt = f"""
You are a company knowledge assistant.

Answer the user's question using ONLY
the document context below.

Rules:

- Give a clear and accurate answer.
- Keep the answer concise.
- Do not invent information.
- Use only the supplied document information.

Document context:
{context}

Question:
{question}

Answer:
"""


        with st.spinner(
            "Searching knowledge base..."
        ):

            response = llm.invoke(
                answer_prompt
            )


        answer = response.content


        # Create source text
        source_lines = []

        for file_name, page in sources:

            source_lines.append(
                f"{file_name} — Page {page}"
            )


        source_text = "<br>".join(
            source_lines
        )


        final_answer = (
            f"{answer}\n\n"
            f"**📄 Source:**<br>{source_text}"
        )


    # ---------------------------------
    # General AI answer
    # ---------------------------------
    else:

        history = ""

        for message in (
            st.session_state.messages[-6:]
        ):

            history += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )


        general_prompt = f"""
You are a friendly and helpful AI assistant.

Answer the user's question naturally
using your general knowledge.

Do not pretend the answer came from
the company documents.

If the user says hello, thanks, ok,
bye, or similar casual messages,
respond naturally.

Keep answers clear and concise.

Recent conversation:
{history}

User:
{question}

Assistant:
"""


        with st.spinner("Thinking..."):

            response = llm.invoke(
                general_prompt
            )


        final_answer = response.content


    # ---------------------------------
    # Display answer
    # ---------------------------------
    with st.chat_message("assistant"):

        st.markdown(
            final_answer,
            unsafe_allow_html=True
        )


    # Save assistant answer
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer
        }
    )