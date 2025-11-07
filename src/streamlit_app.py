"""
Streamlit interface for RAG System with Conversation Memory
Simple Q&A interface using pre-processed PDFs from test files folder
"""

import os
import streamlit as st
from src.langchain_rag import (
    load_t5_summarizer,
    process_pdf,
    create_embeddings,
    create_vector_store,
    setup_rag_chain
)

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "summaries" not in st.session_state:
    st.session_state.summaries = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


def get_pdf_files():
    """Get list of PDF files from test files folder."""
    pdf_dir = "test files"
    if os.path.exists(pdf_dir):
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) 
                    if f.endswith('.pdf')]
        return pdf_files
    return []


def process_pdfs():
    """Process all PDFs and create RAG chain."""
    pdf_files = get_pdf_files()
    
    if not pdf_files:
        st.error("No PDF files found in 'test files' folder!")
        return False
    
    try:
        # Load T5 summarizer
        with st.spinner("Loading T5 model..."):
            summarizer = load_t5_summarizer()
        
        # Process PDFs
        summaries = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, pdf_path in enumerate(pdf_files):
            status_text.text(f"Processing {os.path.basename(pdf_path)} ({idx+1}/{len(pdf_files)})...")
            try:
                summary = process_pdf(pdf_path, summarizer)
                summaries.append(summary)
            except Exception as e:
                st.error(f"Error processing {pdf_path}: {e}")
                continue
            progress_bar.progress((idx + 1) / len(pdf_files))
        
        if not summaries:
            st.error("No summaries generated. Please check PDF files.")
            return False
        
        # Create embeddings and vector store
        with st.spinner("Creating embeddings and vector store..."):
            embeddings = create_embeddings()
            vector_store = create_vector_store(summaries, embeddings)
        
        # Setup RAG chain
        with st.spinner("Setting up RAG chain..."):
            qa_chain = setup_rag_chain(vector_store, embeddings)
        
        # Store in session state
        st.session_state.summaries = summaries
        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
        
        status_text.empty()
        progress_bar.empty()
        return True
        
    except Exception as e:
        st.error(f"Error during processing: {e}")
        return False


def clear_conversation():
    """Clear conversation history."""
    if st.session_state.qa_chain:
        try:
            st.session_state.qa_chain.memory.clear()
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
        except Exception as e:
            st.error(f"Error clearing conversation: {e}")


# Main UI
st.title("📚 RAG Q&A System")
st.markdown("Ask questions about your research papers using AI-powered retrieval")

# Sidebar for PDF processing
with st.sidebar:
    st.header("📄 PDF Processing")
    
    pdf_files = get_pdf_files()
    if pdf_files:
        st.write(f"Found {len(pdf_files)} PDF file(s):")
        for pdf_file in pdf_files:
            st.write(f"- {os.path.basename(pdf_file)}")
    else:
        st.warning("No PDF files found in 'test files' folder")
    
    st.divider()
    
    if st.button("🔄 Process PDFs", type="primary", use_container_width=True):
        if pdf_files:
            if process_pdfs():
                st.success("PDFs processed successfully! You can now ask questions.")
        else:
            st.error("No PDF files to process!")
    
    st.divider()
    
    if st.session_state.qa_chain:
        st.success("✅ System ready")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            clear_conversation()
    else:
        st.info("Please process PDFs first to start Q&A")

# Main Q&A interface
if st.session_state.qa_chain:
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("💬 Conversation History")
        for i, (question, answer) in enumerate(st.session_state.conversation_history):
            with st.expander(f"Q: {question[:50]}...", expanded=(i == len(st.session_state.conversation_history) - 1)):
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {answer}")
    
    # Question input
    st.subheader("❓ Ask a Question")
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main contribution of this paper?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Process question
    if ask_button and question:
        try:
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"question": question})
                answer = result.get("answer", "No answer generated.")
                
                # Store in conversation history
                st.session_state.conversation_history.append((question, answer))
                
                # Display answer
                st.success("Answer:")
                st.write(answer)
                
                # Show source documents count
                source_docs = result.get("source_documents", [])
                if source_docs:
                    st.info(f"Retrieved {len(source_docs)} relevant document(s)")
                
                # Rerun to update conversation history display
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing question: {e}")
    
    elif ask_button and not question:
        st.warning("Please enter a question first.")
        
else:
    st.info("👈 Please process PDFs from the sidebar to start asking questions.")
    st.markdown("""
    ### Instructions:
    1. Place your PDF files in the `test files` folder
    2. Click "Process PDFs" button in the sidebar
    3. Wait for processing to complete
    4. Start asking questions!
    """)

