from pydantic import BaseModel
from typing import List, Dict, Optional

class Document(BaseModel):
    id: str
    filename: str
    file_path: str
    content: str

class DocumentChunk(BaseModel):
    document_id: str
    text: str
    metadata: Dict = {}

class DocumentResponse(BaseModel):
    document_id: str
    extracted_answer: str
    citation: str

class ThemeResponse(BaseModel):
    theme_id: str
    theme_name: str
    description: str
    supporting_documents: List[str]

class QueryResponse(BaseModel):
    query: str
    document_responses: List[DocumentResponse]
    themes: List[ThemeResponse]