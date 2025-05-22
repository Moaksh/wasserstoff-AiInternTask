import streamlit as st
import requests
import os
import json
from typing import List, Dict, Any
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Document Research & Theme Identification Chatbot",
    page_icon="📚",
    layout="wide"
)

# API endpoint (when running locally)
API_URL = "http://localhost:8000/api/v1"

# Initialize session state variables
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'selected_docs' not in st.session_state:
    st.session_state.selected_docs = []

# Load existing documents from API
def load_existing_documents():
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            documents = response.json()
            if documents:
                # Store documents with their processing status
                for doc in documents:
                    # Set default processing status if not provided
                    if 'status' not in doc:
                        doc['status'] = 'unknown'
                
                st.session_state.uploaded_docs = documents
                return True
        return False
    except Exception as e:
        st.error(f"Error loading existing documents: {str(e)}")
        return False

# Try to load existing documents when app starts
if not st.session_state.uploaded_docs:
    load_existing_documents()

# Title and description
st.title("📚 Document Research & Theme Identification Chatbot")
st.markdown(
    """Upload documents, ask questions, and discover common themes across your document collection.
    This application processes your documents, extracts information, and identifies thematic connections."""
)

# Create sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # Add refresh button to reload documents from API
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Document List"):
            if load_existing_documents():
                st.success("Document list refreshed successfully!")
            else:
                st.warning("No documents found or unable to connect to API.")
    
    # Add delete all documents button
    with col2:
        if st.button("🗑️ Delete All Documents"):
            try:
                response = requests.delete(f"{API_URL}/delete")
                if response.status_code == 200:
                    st.session_state.uploaded_docs = []
                    st.success("All documents deleted successfully!")
                else:
                    st.error(f"Error deleting documents: {response.text}")
            except Exception as e:
                st.error(f"Error deleting documents: {str(e)}")

    # Add delete selected documents button
    selected_docs_to_delete = st.multiselect(
        "Select documents to delete",
        options=[doc['doc_id'] for doc in st.session_state.uploaded_docs],
        format_func=lambda x: f"{x} - {next(doc['filename'] for doc in st.session_state.uploaded_docs if doc['doc_id'] == x)}"
    )

    if st.button("🗑️ Delete Selected Documents"):
        if selected_docs_to_delete:
            try:
                for doc_id in selected_docs_to_delete:
                    response = requests.delete(f"{API_URL}/delete/{doc_id}")
                    if response.status_code == 200:
                        # Remove deleted document from session state
                        st.session_state.uploaded_docs = [doc for doc in st.session_state.uploaded_docs if doc['doc_id'] != doc_id]
                    else:
                        st.error(f"Error deleting document {doc_id}: {response.text}")
                st.success("Selected documents deleted successfully!")
            except Exception as e:
                st.error(f"Error deleting documents: {str(e)}")

    
    # Document upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, image, or text files (75+ documents recommended)", 
        accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "txt"]
    )
    
    if uploaded_files and st.button("Process Documents"):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        for i, file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
            
            # Save file temporarily
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Upload to API
            try:
                files = {"file": (file.name, open(temp_path, "rb"))}
                response = requests.post(f"{API_URL}/upload", files=files)
                
                if response.status_code == 201:
                    result = response.json()
                    st.session_state.uploaded_docs.append(result)
                else:
                    st.error(f"Error uploading {file.name}: {response.text}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Clear progress bar and show success message
        progress_bar.empty()
        status_text.empty()
        st.success(f"Successfully processed {len(uploaded_files)} documents!")
    
    # Display uploaded documents
    st.subheader("Uploaded Documents")
    if st.session_state.uploaded_docs:
        # Create a more detailed dataframe
        doc_data = []
        for doc in st.session_state.uploaded_docs:
            doc_info = {
                'ID': doc['doc_id'],
                'Filename': doc['filename'],
                'Status': doc.get('status', 'unknown')
            }
            doc_data.append(doc_info)
        
        # Display the dataframe
        doc_df = pd.DataFrame(doc_data)
        st.dataframe(doc_df, use_container_width=True)
        
        # Show document count
        st.info(f"Total documents: {len(st.session_state.uploaded_docs)}")
        
        # Option to select specific documents for querying
        st.subheader("Filter Documents for Query")
        # Only show documents with status 'ready' in the selection
        queryable_docs = [doc['doc_id'] for doc in st.session_state.uploaded_docs if doc.get('status', '') == 'ready']
        selected_docs = st.multiselect(
            "Select documents to include in query (leave empty to query all)",
            options=queryable_docs,
            format_func=lambda x: f"{x} - {next(doc['filename'] for doc in st.session_state.uploaded_docs if doc['doc_id'] == x)}"
        )
        st.session_state.selected_docs = selected_docs
        
        # Warning if no queryable documents
        if not queryable_docs:
            st.warning("⚠️ No documents are ready for querying yet. Please wait for processing to complete.")

# Main content area
st.header("Ask Questions & Discover Themes")

# Query input
query = st.text_input("Enter your question about the documents:")

# Check if there are any queryable documents
queryable_docs_exist = any(doc.get('status', '') == 'ready' for doc in st.session_state.uploaded_docs)

# Disable query button if no queryable documents
query_button = st.button("Submit Query", disabled=not queryable_docs_exist)

if not queryable_docs_exist:
    st.warning("⚠️ No documents are ready for querying yet. Please wait for processing to complete.")

if query and query_button:
    if not st.session_state.uploaded_docs:
        st.warning("Please upload documents before submitting a query.")
    else:
        with st.spinner("Processing your query..."):
            try:
                # Prepare data for API request
                data = {
                    "query": query
                }
                
                # Add selected documents if any
                if st.session_state.selected_docs:
                    data["doc_id"] = st.session_state.selected_docs[0]  # For single document query
                
                # Send query to API
                response = requests.post(f"{API_URL}/query", json=data)
                
                if response.status_code == 200:
                    st.session_state.query_results = response.json()
                else:
                    st.error(f"Error processing query: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Display query results
if st.session_state.query_results:
    results = st.session_state.query_results
    
    # Display direct answer if available
    if 'direct_answer' in results and results['direct_answer']:
        st.subheader("Direct Answer")
        st.write(results['direct_answer'])
    
    # Display individual document responses in a table
    st.subheader("Individual Document Responses")
    
    # Create a DataFrame for document responses
    if 'sources' in results and results['sources']:
        doc_responses = []
        for source in results['sources']:
            doc_responses.append({
                "Document ID": source["doc_id"],
                "Extracted Answer": source["content_preview"],
                "Citation": f"Page {source.get('page_number', 'N/A')}"
            })
        
        if doc_responses:
            st.table(pd.DataFrame(doc_responses))
        else:
            st.info("No relevant information found in documents.")
    else:
        st.info("No relevant information found in documents.")
    
    # Display identified themes
    if 'identified_themes' in results and results['identified_themes']:
        st.subheader("Identified Themes")
        
        for theme in results['identified_themes']:
            with st.expander(f"Theme: {theme['theme_title']}"):
                st.markdown(f"**Description:** {theme['summary']}")
                st.markdown("**Supporting Documents:**")
                for doc_id in theme["supporting_doc_ids"]:
                    st.markdown(f"- {doc_id}")
    else:
        st.info("No themes identified across documents.")

# Add information about the application
st.markdown("---")
st.markdown("""
### About this Application

This Document Research & Theme Identification Chatbot uses advanced AI to analyze documents, 
identify themes, and provide cited responses to your questions.

**Features:**
- Upload and process multiple document formats (PDF, images, text)
- Extract text using OCR for scanned documents
- Query documents with natural language questions
- Receive answers with precise citations
- Discover common themes across your document collection

**Technologies:**
- Google Gemini AI for natural language processing
- ChromaDB for vector storage and semantic search
- OCR for text extraction from images and PDFs
""")