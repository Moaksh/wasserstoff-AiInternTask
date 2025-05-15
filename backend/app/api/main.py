import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import shutil
import uuid

# Import our custom modules
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore
from services.theme_identifier import ThemeIdentifier
from models.models import Document, DocumentResponse, ThemeResponse
from config import HOST, PORT, DEBUG, UPLOAD_FOLDER, VECTOR_DB_FOLDER

# Initialize FastAPI app
app = FastAPI(title="Document Research & Theme Identification Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
vector_store = VectorStore(VECTOR_DB_FOLDER)
theme_identifier = ThemeIdentifier()

# API routes
@app.get("/")
def read_root():
    return {"message": "Document Research & Theme Identification Chatbot API"}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    uploaded_docs = []
    for file in files:
        # Generate unique ID for the document
        doc_id = f"DOC{str(uuid.uuid4())[:6]}"
        
        # Save the file
        file_path = f"{UPLOAD_FOLDER}/{doc_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document (extract text, perform OCR if needed)
        processed_text = document_processor.process_document(file_path)
        
        # Create document object
        doc = Document(
            id=doc_id,
            filename=file.filename,
            file_path=file_path,
            content=processed_text
        )
        
        # Add to vector store
        vector_store.add_document(doc)
        
        uploaded_docs.append({
            "id": doc_id,
            "filename": file.filename,
            "size": os.path.getsize(file_path)
        })
    
    return {"message": f"Successfully uploaded {len(uploaded_docs)} documents", "documents": uploaded_docs}

@app.get("/documents")
def get_documents():
    # Retrieve all documents from vector store
    docs = vector_store.get_all_documents()
    return {"documents": docs}

@app.delete("/documents")
def delete_all_documents():
    # Delete all documents from vector store
    result = vector_store.clear()
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to clear vector store")
    
    # Delete all files in upload folder
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return {"message": "All documents deleted successfully"}

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    # Delete document from vector store
    result = vector_store.delete_document(doc_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found or could not be deleted")
    
    # Delete the file from upload folder
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.startswith(f"{doc_id}_"):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    return {"message": f"Document {doc_id} deleted successfully"}

@app.post("/query")
async def query_documents(query: str = Form(...), doc_ids: Optional[List[str]] = Form(None)):
    # If doc_ids is provided, only query those documents
    # Otherwise, query all documents
    
    # Get relevant document chunks from vector store
    results = vector_store.query(query, doc_ids)
    
    # Process each document to extract answers with citations
    document_responses = []
    for doc_id, chunks in results.items():
        response = document_processor.extract_answer(query, chunks)
        document_responses.append(DocumentResponse(
            document_id=doc_id,
            extracted_answer=response["answer"],
            citation=response["citation"]
        ))
    
    # Identify themes across document responses
    themes = theme_identifier.identify_themes(query, document_responses)
    
    # Format the final response
    theme_responses = []
    for theme_id, theme_data in themes.items():
        theme_responses.append(ThemeResponse(
            theme_id=theme_id,
            theme_name=theme_data["name"],
            description=theme_data["description"],
            supporting_documents=theme_data["supporting_documents"]
        ))
    
    return {
        "query": query,
        "document_responses": document_responses,
        "themes": theme_responses
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=DEBUG)