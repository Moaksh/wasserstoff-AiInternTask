from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DocumentStatus(BaseModel):
    doc_id: str
    filename: str
    status: str
    message: Optional[str] = None

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query for the document(s).")
    doc_id: Optional[str] = Field(None, description="Optional specific document ID to query. Queries all documents if None.")
    top_k: int = Field(default=5, ge=1, le=15, description="Number of relevant chunks to retrieve for analysis.") # Using default from config.py can be an option too

class IdentifiedTheme(BaseModel):
    theme_title: str
    supporting_doc_ids: List[str] = Field(default_factory=list)
    summary: str

class QueryResponse(BaseModel):
    direct_answer: Optional[str] = None # This will be the overall synthesized answer
    identified_themes: Optional[List[IdentifiedTheme]] = None
    sources: List[Dict[str, Any]] # All unique source chunks retrieved 
    message: Optional[str] = None

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    message: str
    status: str

class DeletionResponse(BaseModel):
    message: str
    deleted_doc_ids: Optional[List[str]] = None
    errors: Optional[List[str]] = None

