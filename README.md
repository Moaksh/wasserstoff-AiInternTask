# Document Research & Theme Identification System

This project is a document research and theme identification system that allows users to upload PDF documents, query their content, and discover thematic connections across documents. The system uses AI to extract information from documents and identify common themes.

## Features

- **Document Upload**: Upload and process PDF documents
- **Document Management**: View, filter, and delete documents
- **Semantic Search**: Query documents using natural language
- **Theme Identification**: Automatically identify themes across documents
- **Direct Answers**: Get concise answers to specific questions

## Project Structure

- **Backend**: FastAPI application with modular services
- **Frontend**: Streamlit web interface
- **Vector Database**: ChromaDB for document embeddings and semantic search
- **LLM Integration**: Google Generative AI for text processing and theme identification

## Prerequisites

- Python 3.8+
- Google API Key for Generative AI services

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Moaksh/wasserstoff-AiInternTask
   cd wasserstoff-AiInternTask
   ```

2. Create and activate a Conda environment (optional but recommended):
   ```bash
   conda create -n wasserstoff python=3.8
   conda activate wasserstoff
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

## Running the Application

### Backend

1. Start the FastAPI backend:
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```
   The API will be available at http://localhost:8000
   - API documentation: http://localhost:8000/docs

### Frontend

1. Start the Streamlit frontend:
   ```bash
   cd demo
   streamlit run app.py
   ```
   The web interface will be available at http://localhost:8501

**Note on Hosting:** Due to the system storing uploaded PDF documents, hosting on free-tier services is not feasible. Local deployment is recommended.


## Usage

1. Open the Streamlit interface in your browser
2. Upload PDF documents using the sidebar
3. Wait for documents to be processed (status will change to "ready")
4. Enter questions in the query box
5. View direct answers, identified themes, and source documents

## Solution Approach

### Document Processing Pipeline

1. **Document Upload**: PDFs are uploaded and stored locally
2. **Text Extraction**: Text is extracted from PDFs using pdfplumber
3. **Chunking**: Documents are split into manageable chunks with overlap
4. **Embedding**: Text chunks are embedded using Google's embedding model
5. **Vector Storage**: Embeddings are stored in ChromaDB for semantic search

### Query Processing Pipeline

1. **Query Embedding**: User queries are embedded using the same model
2. **Semantic Search**: ChromaDB finds relevant document chunks
3. **Context Building**: Relevant chunks are assembled into context
4. **LLM Processing**: Google's Generative AI processes the context to:
   - Generate direct answers to user queries
   - Identify themes across documents
   - Provide source citations

### Theme Identification

The theme identification system works by:

1. Analyzing semantic relationships between document chunks
2. Using prompt engineering to guide the LLM in identifying themes
3. Parsing structured theme data from LLM responses
4. Organizing themes with supporting document references

## API Endpoints

- `POST /api/v1/upload`: Upload a PDF document
- `GET /api/v1/documents`: List all documents
- `DELETE /api/v1/delete/{doc_id}`: Delete a specific document
- `DELETE /api/v1/delete`: Delete all documents
- `POST /api/v1/query`: Query documents with natural language

## Technologies Used

- **FastAPI**: Backend web framework
- **Streamlit**: Frontend web interface
- **ChromaDB**: Vector database for semantic search
- **Langchain**: Framework for LLM applications
- **Google Generative AI**: LLM and embedding models
- **PDFPlumber**: PDF text extraction

## Challenges Faced

- **Gemma API Limitations**: The Gemma API has certain rate limits and token constraints that affected the ability to provide detailed individual responses for each document
- **Context Window Limitations**: Working with large documents required careful chunking to fit within LLM context windows
- **Parsing Structured Outputs**: Extracting structured theme data from LLM responses required robust regex parsing
- **Vector Database Optimization**: Tuning ChromaDB for optimal semantic search performance was challenging
- **Error Handling**: Managing various failure modes in the document processing pipeline required comprehensive error handling
- **Scalability**: The system's scalability was limited by the API's rate limits and the need for fine-tuned chunking and embedding models
- **Hosting**: Hosting the system on free-tier services was not feasible due to the storage of uploaded PDFs

