"""
LangChain-based RAG System with Conversation Memory
Uses fine-tuned T5 model for summarization and LangChain for RAG pipeline
"""

import os
import re
from dotenv import load_dotenv
from typing import List, Optional

# PDF Processing
import pdfplumber

# Summarization
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Load environment variables
load_dotenv()


# ============================================================================
# PDF Processing & Summarization Functions
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")
    return text


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = re.sub(r'\\[a-z]+', '', text)  # Remove escape sequences
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
    """Split text into chunks with overlap (sliding window approach).
    
    Uses smaller chunk size to avoid token limit issues and reduce processing time.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap  # Step size for sliding window
    for i in range(0, len(words), step):
        chunk = words[i:i+chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(" ".join(chunk))
    return chunks


def load_t5_summarizer(model_path: str = "./models/t5_arxiv_full_model"):
    """Load fine-tuned T5 model and tokenizer for summarization."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")
        return summarizer
    except Exception as e:
        raise Exception(f"Error loading T5 model: {e}")


def summarize_chunks(summarizer, chunks: List[str], max_length: int = 150, min_length: int = 40) -> str:
    """Summarize chunks and merge into a single summary.
    
    Note: Processing is sequential due to CPU inference bottleneck.
    Each chunk takes ~5-10 seconds on CPU.
    """
    summaries = []
    for i, chunk in enumerate(chunks):
        input_text = "summarize: " + chunk
        try:
            summary = summarizer(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            raise Exception(f"Error summarizing chunk {i+1}: {e}")
    
    return " ".join(summaries)


def process_pdf(pdf_path: str, summarizer) -> str:
    """Process PDF: extract, clean, chunk, and summarize."""
    try:
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)
        chunks = split_into_chunks(cleaned_text, chunk_size=200, overlap=30)
        merged_summary = summarize_chunks(summarizer, chunks)
        return merged_summary
    except Exception as e:
        raise Exception(f"Error processing PDF {pdf_path}: {e}")


# ============================================================================
# LangChain RAG Setup
# ============================================================================

def create_embeddings():
    """Create HuggingFace embeddings using sentence-transformers."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return embeddings
    except Exception as e:
        raise Exception(f"Error creating embeddings: {e}")


def create_vector_store(summaries: List[str], embeddings) -> FAISS:
    """Create FAISS vector store from summaries."""
    try:
        documents = [Document(page_content=summary, metadata={"index": i}) 
                     for i, summary in enumerate(summaries)]
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        raise Exception(f"Error creating vector store: {e}")


def create_prompt_templates():
    """Create custom prompt templates for the RAG chain."""
    
    # QA Prompt Template
    qa_template = """You are an expert research assistant that helps answer questions about academic papers.
You have access to summarized context from research papers. Answer questions based ONLY on the provided context.

If the answer cannot be found in the context, say "I cannot find this information in the provided context."

Context: {context}

Question: {question}

Answer based on the context above:"""
    
    QA_PROMPT = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question"]
    )
    
    # Condense Question Template (for follow-up questions)
    condense_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that incorporates context from the conversation history.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate(
        template=condense_template,
        input_variables=["chat_history", "question"]
    )
    
    return QA_PROMPT, CONDENSE_QUESTION_PROMPT


def setup_rag_chain(vector_store: FAISS, embeddings):
    """Set up the conversational RAG chain with memory."""
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables!")
    
    try:
        # Create LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=500
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
        )
        
        # Create conversation memory with summary
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create prompt templates
        QA_PROMPT, CONDENSE_QUESTION_PROMPT = create_prompt_templates()
        
        # Create conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            verbose=False,
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        raise Exception(f"Error setting up RAG chain: {e}")


# ============================================================================
# Interactive Interface
# ============================================================================

def interactive_qa(qa_chain):
    """Interactive Q&A loop."""
    print("\n" + "="*60)
    print("Interactive Q&A Session Started")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'clear' to clear conversation history")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user query
            query = input("\nQuestion: ").strip()
            
            if not query:
                continue
            
            # Handle exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nEnding session. Goodbye!")
                break
            
            # Handle clear command
            if query.lower() == 'clear':
                qa_chain.memory.clear()
                print("Conversation history cleared!")
                continue
            
            # Get answer from chain
            print("\nThinking...")
            result = qa_chain.invoke({"question": query})
            
            # Display answer
            answer = result.get("answer", "No answer generated.")
            print(f"\nAnswer: {answer}")
            
            # Optionally show source documents
            source_docs = result.get("source_documents", [])
            if source_docs and len(source_docs) > 0:
                print(f"\n[Retrieved {len(source_docs)} relevant document(s)]")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'exit' to quit.")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to run the RAG system."""
    import sys
    sys.stdout.flush()  # Force flush output
    print("="*60, flush=True)
    print("LangChain RAG System with Conversation Memory", flush=True)
    print("="*60, flush=True)
    
    # Step 1: Load T5 summarizer
    try:
        summarizer = load_t5_summarizer()
    except Exception as e:
        print(f"Error loading T5 model: {e}")
        print("Make sure the model is available at ./models/t5_arxiv_full_model/")
        return
    
    # Step 2: Process PDFs
    print("\n" + "-"*60)
    print("PDF Processing")
    print("-"*60)
    
    pdf_files = []
    pdf_dir = "test files"
    
    # Check for PDF files
    if os.path.exists(pdf_dir):
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) 
                    if f.endswith('.pdf')]
    else:
        # Fallback to current directory
        pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found!")
        print("Please place PDF files in the 'test files' directory or current directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Process all PDFs
    summaries = []
    for pdf_path in pdf_files:
        try:
            summary = process_pdf(pdf_path, summarizer)
            summaries.append(summary)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    if not summaries:
        print("No summaries generated. Exiting.")
        return
    
    # Step 3: Create embeddings and vector store
    print("\n" + "-"*60)
    print("Vector Store Setup")
    print("-"*60)
    
    embeddings = create_embeddings()
    vector_store = create_vector_store(summaries, embeddings)
    
    # Step 4: Setup RAG chain
    print("\n" + "-"*60)
    print("RAG Chain Setup")
    print("-"*60)
    
    try:
        qa_chain = setup_rag_chain(vector_store, embeddings)
    except Exception as e:
        print(f"Error setting up RAG chain: {e}")
        return
    
    # Step 5: Start interactive Q&A
    print("\n" + "-"*60)
    print("Ready for Questions!")
    print("-"*60)
    
    interactive_qa(qa_chain)


if __name__ == "__main__":
    try:
        print("Starting script...", flush=True)
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}", flush=True)
        traceback.print_exc()
