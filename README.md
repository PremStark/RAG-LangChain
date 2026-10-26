# RAG using LangChain

A Retrieval-Augmented Generation (RAG) pipeline built step by step in a single Jupyter notebook. It reads PDFs, splits them into chunks, converts the chunks into embeddings, stores them in a vector database, and then answers questions by retrieving the most relevant chunks and passing them to an LLM.

## How it works

```
PDFs → Documents → Chunks → Embeddings → ChromaDB
                                            │
 Question → Embedding → Top-k similar chunks┘
                                            │
                       Context + Question → LLM → Answer
```

1. **Ingestion**
   - Loads every PDF in a folder with `PyPDFLoader`, one LangChain `Document` per page.
   - Splits pages into chunks using `RecursiveCharacterTextSplitter` (chunk size 500, overlap 50).
2. **Embedding**
   - Uses `all-MiniLM-L6-v2` from `sentence-transformers` (384-dimensional vectors) through an `EmbeddingGenerator` class.
3. **Vector store**
   - Stores chunks, embeddings and metadata in a persistent **ChromaDB** collection through a `VectorStoreManager` class.
4. **Retrieval**
   - `RAGRetriever` embeds the query, fetches the top-k closest chunks, and converts distance to a similarity score with an optional threshold.
5. **Generation**
   - Retrieved chunks are joined into a context and sent with the question to an LLM via `langchain-groq` (`ChatGroq`).

## Tech stack

| Part | Tool |
|---|---|
| Framework | LangChain |
| PDF loading | PyPDF, PyMuPDF |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector database | ChromaDB |
| LLM | Groq API (`langchain-groq`) |
| Environment | Jupyter / Google Colab |

## Getting started

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install langchain langchain-core langchain-community langchain_text_splitters \
            pypdf pymupdf sentence-transformers chromadb scikit-learn langchain-groq
```

### 3. Add your PDFs

Create a `data/` folder next to the notebook and put your PDF files inside it.

> The notebook was written in Google Colab, so `load_documents()` points to `/content/data`. If you run it locally, change `folder_path` to `"data"`.

### 4. Set your Groq API key

Get a key from [console.groq.com](https://console.groq.com) and set it in the notebook:

```python
API_KEY = "YOUR_LLM_API_KEY"
```

Do not commit a real key to GitHub. A safer option is to read it from an environment variable:

```python
import os
API_KEY = os.environ["GROQ_API_KEY"]
```

### 5. Run the notebook

Open `RAG-Project.ipynb` and run the cells from top to bottom.

## Example

```python
answer = generate_output("What is Encoder and Decoder ?", rag_retriever, llm)
print(answer)
```

The retriever finds the most relevant chunks from your PDFs, and the LLM answers using only that context.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Chunk size / overlap | `split_docs()` | 500 / 50 |
| Embedding model | `EmbeddingGenerator` | `all-MiniLM-L6-v2` |
| Vector store path | `VectorStoreManager` | `data/vector_store` |
| Collection name | `VectorStoreManager` | `pdf_Docs` |
| Chunks retrieved | `generate_output(top_k=...)` | 3 |
| LLM model / temperature | `ChatGroq(...)` | set in notebook / 0.1 |

## Possible improvements

- Add a reranker for better retrieval quality.
- Support more file types (DOCX, web pages).
- Show source file and page number alongside each answer.
- Wrap the pipeline in a simple Streamlit or Gradio chat UI.
