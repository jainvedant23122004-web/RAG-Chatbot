from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = PROJECT_ROOT / "docs"
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents():
    documents = []

    for file_path in DOCS_PATH.rglob("*.txt"):
        documents.extend(
            TextLoader(str(file_path), encoding="utf-8").load()
        )

    for file_path in DOCS_PATH.rglob("*.pdf"):
        documents.extend(PyPDFLoader(str(file_path)).load())

    return documents


def main():
    if not DOCS_PATH.exists():
        print(f"Documents folder not found: {DOCS_PATH}")
        return

    documents = load_documents()

    if not documents:
        print("No PDF or TXT files found in the docs folder.")
        print(f"Add files to: {DOCS_PATH}")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH),
	collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Loaded documents/pages: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Saved ChromaDB index to: {DB_PATH}")


if __name__ == "__main__":
    main()