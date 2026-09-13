\# RAG Chatbot with Groq



A local Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF and text documents.



\## Features



\- Loads PDF and TXT documents from the `docs/` folder

\- Splits documents into smaller chunks

\- Generates local embeddings using Sentence Transformers

\- Stores embeddings in ChromaDB

\- Retrieves relevant chunks for each question

\- Uses Groq's Llama 3.1 8B model for fast response generation

\- Shows source document and page information with answers



\## Tech Stack



\- Python

\- LangChain

\- ChromaDB

\- Sentence Transformers

\- Groq API

\- Llama 3.1 8B Instant

\- PyPDF



\## Project Structure



```text

RAG-chatbot/

├── docs/              # Local source documents (not committed)

├── src/

│   ├── test\_groq.py   # Groq API connectivity test

│   ├── ingest.py      # Document ingestion and vector indexing

│   └── rag.py         # Terminal chatbot

├── .env               # Groq API key (not committed)

├── .gitignore

├── requirements.txt

└── README.md

```



\## Setup



1\. Create and activate a Python virtual environment.

2\. Install dependencies:



&#x20;  ```powershell

&#x20;  python -m pip install -r requirements.txt

&#x20;  ```



3\. Create a `.env` file in the project root:



&#x20;  ```env

&#x20;  GROQ\_API\_KEY=your\_groq\_api\_key\_here

&#x20;  ```



4\. Add PDF or TXT files to `docs/`.

5\. Run ingestion:



&#x20;  ```powershell

&#x20;  python src/ingest.py

&#x20;  ```



6\. Start the chatbot:



&#x20;  ```powershell

&#x20;  python src/rag.py

&#x20;  ```



\## Security



Never commit the `.env` file. It contains the Groq API key and is excluded through `.gitignore`.

