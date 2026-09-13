import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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


def show_sources(documents):
    print("\nSources used:")

    for number, document in enumerate(documents, start=1):
        source = Path(document.metadata.get("source", "Unknown")).name
        page = document.metadata.get("page")

        if page is not None:
            print(f"{number}. {source} - page {page + 1}")
        else:
            print(f"{number}. {source}")


def main():
    load_dotenv()

    print("Starting RAG chatbot...")

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY was not found. Check your .env file.")
        return

    if not DB_PATH.exists():
        print("ChromaDB index not found.")
        print("Run this first: python .\\src\\ingest.py")
        return

    print("Loading local embedding model...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH),
        embedding_function=get_embeddings(),
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    print("Connecting to Groq...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    print("\nRAG chatbot is ready.")
    print("Ask a question, or type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Chatbot: Goodbye.")
            break

        if not question:
            continue

        print("Searching documents...")

        documents = retriever.invoke(question)

        if not documents:
            print(
                "\nChatbot: I could not find that information "
                "in the provided documents.\n"
            )
            continue

        context = format_context(documents)

        print("Generating answer...")

        response = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        print(f"\nChatbot: {response.content}")
        show_sources(documents)
        print()


if __name__ == "__main__":
    main()