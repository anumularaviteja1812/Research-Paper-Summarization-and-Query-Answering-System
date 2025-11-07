#  Summarization and Query Answering System

This project involves building an end-to-end system for summarizing academic papers from the Arxiv dataset and answering questions based on the generated summaries. It includes the following key components:

## Features
- **Data Preprocessing**: Load and preprocess the Arxiv summarization dataset using Hugging Face's `datasets` library.
- **Model Fine-Tuning**: Fine-tune the `T5-small` model for abstractive summarization.
- **PDF Text Extraction**: Extract text from PDFs using `pdfplumber` and preprocess it for summarization.
- **Summarization Pipeline**: Generate summaries from text chunks using a fine-tuned T5 model.
- **Semantic Search**: Use FAISS to index and retrieve relevant summaries based on a query.
- **Interactive Query Answering**: Use OpenAI's GPT-3.5 to generate answers based on retrieved summaries.

## Project Structure

```
Summarizer project/
├── src/                          # Source code
│   ├── langchain_rag.py         # LangChain-based RAG implementation
│   └── streamlit_app.py          # Streamlit web interface
│
├── notebooks/                    # Jupyter notebooks
│   ├── Rag.ipynb                 # Normal RAG implementation (without LangChain)
│   └── t5_train.ipynb            # T5 model training notebook
│
├── data/                         # Processed data files
│   └── final.csv
│
├── models/                       # Fine-tuned models
│   └── t5_arxiv_full_model/     # Fine-tuned T5 model for summarization
│
└── test files/                   # Test PDF files
    ├── 0.pdf
    └── 1.pdf
```

## Two Implementation Approaches

This project provides **two different RAG (Retrieval-Augmented Generation) implementations**:

### 1. **Normal RAG Implementation** (without LangChain)
- **Location**: `notebooks/Rag.ipynb`
- **Features**:
  - Direct `SentenceTransformer` ("all-MiniLM-L6-v2") for embeddings (not LangChain wrapper)
  - Direct FAISS index creation using `faiss.IndexFlatL2` (L2 distance)
  - Manual retrieval function `retrieve_summary()` for top-k search
  - Manual prompt construction with string formatting
  - Direct OpenAI API calls (not LangChain's ChatOpenAI)
  - Simple interactive loop without conversation memory
  - Processes PDFs from CSV file (`final.csv`)
- **Use Case**: For learning purposes, understanding the core RAG concepts, or when you want minimal dependencies and full control over the pipeline

### 2. **LangChain-based RAG Implementation**
- **Location**: `src/langchain_rag.py` and `src/streamlit_app.py`
- **Features**:
  - Uses LangChain's `FAISS` vector store wrapper
  - Built-in conversation memory with `ConversationalRetrievalChain`
  - Automatic prompt management with `PromptTemplate`
  - Conversation summary memory for better context handling
  - Streamlit web interface for easy interaction
  - More structured and production-ready code
- **Use Case**: For production applications, when you need conversation memory, or when building web interfaces

### Key Differences

| Feature | Normal RAG | LangChain RAG |
|---------|-----------|---------------|
| **Framework** | Direct libraries (sentence-transformers, faiss, openai) | LangChain framework |
| **Embeddings** | `SentenceTransformer` directly | LangChain's `HuggingFaceEmbeddings` |
| **Vector Store** | `faiss.IndexFlatL2` directly | LangChain's `FAISS.from_documents()` |
| **Retrieval** | Manual `retrieve_summary()` function | LangChain's `as_retriever()` |
| **LLM** | Direct OpenAI API calls | LangChain's `ChatOpenAI` |
| **Prompt Management** | Manual string formatting | `PromptTemplate` with structured templates |
| **Conversation Memory** | None (stateless) | Built-in `ConversationSummaryMemory` |
| **RAG Chain** | Manual `answer_query()` function | `ConversationalRetrievalChain` |
| **Interface** | Jupyter Notebook interactive loop | Command-line + Streamlit web UI |
| **Data Source** | CSV file (`final.csv`) | PDF files from `test files/` folder |
| **Complexity** | Lower (more manual control) | Higher (more features, abstraction) |

## Installation

### Requirements
To run this project, you'll need the following libraries:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install datasets transformers pdfplumber sentence-transformers faiss-cpu openai spacy python-dotenv langchain langchain-community langchain-openai langchain-huggingface streamlit torch
```

Make sure to install the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Running LangChain RAG (Command-line)
```bash
python src/langchain_rag.py
```

### Running LangChain RAG (Streamlit Web Interface)
```bash
streamlit run src/streamlit_app.py
```

### Running Normal RAG
Open `notebooks/Rag.ipynb` in Jupyter Notebook and run the cells.

## Configuration

1. Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

2. Place your PDF files in the `test files/` directory

3. Make sure the fine-tuned T5 model is available at `models/t5_arxiv_full_model/`
