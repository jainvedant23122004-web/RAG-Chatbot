import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """
You are a helpful document question-answering assistant.

Answer the user's question using only the provided context.
Do not use outside knowledge.

If the answer cannot be found in the context, respond exactly:
I could not find that information in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="centered",
)


@st.cache_resource
def load_rag_components():
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY was not found. Add it to the .env file."
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "ChromaDB index not found. Run: python src/ingest.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    return vector_store, chain


def format_context(documents):
    sections = []

    for number, document in enumerate(documents, start=1):
        source = Path(document.metadata.get("source", "Unknown")).name
        page = document.metadata.get("page")
        page_label = f", page {page + 1}" if page is not None else ""

        sections.append(
            f"[Source {number}: {source}{page_label}]\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(sections)


def get_sources(documents):
    sources = []

    for document in documents:
        source = Path(document.metadata.get("source", "Unknown")).name
        page = document.metadata.get("page")

        if page is not None:
            sources.append(f"{source} — page {page + 1}")
        else:
            sources.append(source)

    return list(dict.fromkeys(sources))


def main():
    st.title("📄 RAG Document Chatbot")
    st.caption(
        "Ask questions about documents indexed in the local ChromaDB database."
    )

    with st.sidebar:
        st.header("Project details")
        st.write("**Embeddings:** all-MiniLM-L6-v2 (local)")
        st.write("**Vector database:** ChromaDB (local)")
        st.write("**LLM:** Groq")
        st.divider()
        st.info(
            "To index new PDF or TXT files, add them to the docs folder "
            "and run `python src/ingest.py` in PowerShell."
        )

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    try:
        vector_store, chain = load_rag_components()
    except Exception as error:
        st.error(str(error))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources used"):
                    for source in message["sources"]:
                        st.write(f"- {source}")

    question = st.chat_input("Ask a question about your documents")

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            documents = vector_store.similarity_search(question, k=3)

            if not documents:
                answer = (
                    "I could not find that information in the "
                    "provided documents."
                )
                sources = []
            else:
                context = format_context(documents)

                response = chain.invoke(
                    {
                        "context": context,
                        "question": question,
                    }
                )

                answer = response.content
                sources = get_sources(documents)

        st.success(answer)

        if sources:
            with st.expander("Sources used"):
                for source in sources:
                    st.write(f"- {source}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    main()