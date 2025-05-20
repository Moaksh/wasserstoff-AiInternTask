import os
import uuid
import logging
from typing import List, Optional, Dict, Any

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Body,
    Path,
    Query as FastAPIQuery,
)

from app.models.pydantic_models import (
    DocumentStatus,
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DeletionResponse,
)
from app.services import document_processor as dps
from app.services import vector_store as vss
from app.services import llm_service
from app.core.config import UPLOADED_PDFS_PATH, DEFAULT_QUERY_TOP_K, MAX_QUERY_TOP_K

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/upload", response_model=UploadResponse, status_code=201, tags=["Documents"]
)
async def upload_document_endpoint(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Upload attempt with invalid file: {file.filename or 'N/A'}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file type or missing filename. Only PDF files are accepted.",
        )

    doc_id = str(uuid.uuid4())

    safe_filename_for_path = "".join(
        c if c.isalnum() or c in [".", "_", "-"] else "_" for c in file.filename
    )
    file_location = UPLOADED_PDFS_PATH / f"{doc_id}_{safe_filename_for_path}"

    dps.document_metadata_store[doc_id] = {
        "filename": file.filename,
        "status": "uploading",
        "path": str(file_location),
        "message": "",
    }
    logger.info(f"API: Starting upload for '{file.filename}', doc_id: {doc_id}")

    try:
        await dps.save_uploaded_file_locally(file, str(file_location))

        dps.process_and_store_document(doc_id, file.filename, str(file_location))

        final_status = dps.get_document_status(doc_id)
        if not final_status or final_status.get("status", "").startswith("error"):
            logger.error(
                f"API: Upload processing failed for doc_id {doc_id}. Status: {final_status}"
            )

            raise HTTPException(
                status_code=500,
                detail=final_status.get(
                    "message", "Document processing failed after upload."
                ),
            )

        logger.info(
            f"API: Document {doc_id} processed successfully. Filename: {file.filename}"
        )
        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            message=final_status.get(
                "message", "Document uploaded and processed successfully."
            ),
            status=final_status.get("status", "ready"),
        )
    except HTTPException as http_exc:
        logger.warning(
            f"API: HTTPException during upload for doc_id {doc_id}: {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        logger.error(
            f"API: Unexpected error during upload for doc_id {doc_id}: {e}",
            exc_info=True,
        )
        dps.document_metadata_store[doc_id]["status"] = "error_upload_api"
        dps.document_metadata_store[doc_id]["message"] = f"API level error: {str(e)}"
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred at the API level: {str(e)}",
        )


@router.get("/documents", response_model=List[DocumentStatus], tags=["Documents"])
async def list_documents_endpoint():
    statuses_data = dps.get_all_document_statuses()

    return [DocumentStatus(**data) for data in statuses_data]


@router.delete("/delete/{doc_id}", response_model=DeletionResponse, tags=["Documents"])
async def delete_document_endpoint(
    doc_id: str = Path(..., description="ID of the document to delete.")
):
    logger.info(f"API: Received request to delete document: {doc_id}")
    doc_status = dps.get_document_status(doc_id)
    if not doc_status:
        logger.warning(f"API: Delete failed. Document not found: {doc_id}")
        raise HTTPException(
            status_code=404, detail=f"Document with ID '{doc_id}' not found."
        )

    errors_occurred = []
    try:

        vss.delete_from_vector_store_by_metadata(where_filter={"doc_id": doc_id})
    except Exception as e:
        logger.error(
            f"API: Error deleting doc {doc_id} from vector store: {e}", exc_info=True
        )
        errors_occurred.append(f"Vector store deletion error: {str(e)}")

    if not dps.remove_document_data(doc_id):

        errors_occurred.append(
            "Metadata or local file removal failed (doc might have been partially deleted)."
        )

    if not errors_occurred:
        logger.info(f"API: Document {doc_id} deleted successfully.")
        return DeletionResponse(
            message=f"Document {doc_id} and its data deleted successfully.",
            deleted_doc_ids=[doc_id],
        )
    else:
        logger.warning(
            f"API: Partial deletion for document {doc_id}. Errors: {'; '.join(errors_occurred)}"
        )
        return DeletionResponse(
            message=f"Attempted to delete document {doc_id}. Some operations failed.",
            deleted_doc_ids=[doc_id],
            errors=errors_occurred,
        )


@router.delete("/delete", response_model=DeletionResponse, tags=["Documents"])
async def delete_all_documents_endpoint():
    logger.info("API: Received request to delete ALL documents.")
    errors_occurred = []
    try:
        vss.clear_vector_collection()
    except Exception as e:
        logger.error(f"API: Error clearing vector store collection: {e}", exc_info=True)
        errors_occurred.append(f"Vector store clearing error: {str(e)}")

    deleted_doc_ids = dps.remove_all_documents_data()

    if not errors_occurred:
        logger.info("API: All documents and data deleted successfully.")
        return DeletionResponse(
            message="All documents and their data deleted successfully.",
            deleted_doc_ids=deleted_doc_ids,
        )
    else:
        logger.warning(
            f"API: Partial deletion of all documents. Errors: {'; '.join(errors_occurred)}"
        )
        return DeletionResponse(
            message="Attempted to delete all documents. Some operations failed.",
            deleted_doc_ids=deleted_doc_ids,
            errors=errors_occurred,
        )


@router.post("/query", response_model=QueryResponse, tags=["Querying"])
async def query_documents_endpoint(request: QueryRequest = Body(...)):
    logger.info(
        f"API: Received query: '{request.query[:50]}...', doc_id: {request.doc_id}, top_k: {request.top_k}"
    )

    if not llm_service.llm_instance:
        logger.error("API: LLM service not available for query.")
        raise HTTPException(
            status_code=503, detail="LLM service not available. Check server logs."
        )

    where_filter = None
    if request.doc_id:
        doc_status = dps.get_document_status(request.doc_id)
        if not doc_status:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{request.doc_id}' not found.",
            )
        if doc_status.get("status") != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Document '{doc_status.get('filename')}' (ID: {request.doc_id}) is not ready for querying (status: {doc_status.get('status')}).",
            )
        where_filter = {"doc_id": request.doc_id}

    try:

        query_results = vss.query_vector_store(
            query_texts=[request.query],
            n_results=min(request.top_k, MAX_QUERY_TOP_K),
            where_filter=where_filter,
        )

        if (
            not query_results
            or not query_results.get("documents")
            or not query_results["documents"][0]
        ):
            logger.info("API: No relevant chunks found for query.")
            return QueryResponse(
                message="No relevant information found for your query.", sources=[]
            )

        retrieved_chunk_texts = query_results["documents"][0]
        retrieved_metadatas = query_results["metadatas"][0]

        source_chunks_for_response: List[Dict[str, Any]] = []
        context_parts_for_llm: List[str] = []

        for i, text_chunk in enumerate(retrieved_chunk_texts):
            metadata = retrieved_metadatas[i]
            source_chunks_for_response.append(
                {
                    "doc_id": metadata.get("doc_id", "N/A"),
                    "filename": metadata.get("original_filename", "N/A"),
                    "page_number": metadata.get("page_number", "N/A"),
                    "chunk_index": metadata.get("chunk_index", "N/A"),
                    "content_preview": text_chunk[:150] + "...",
                }
            )
            context_parts_for_llm.append(
                f"Excerpt from Document ID: {metadata.get('doc_id', 'UNKNOWN_DOC_ID')}, Filename: {metadata.get('original_filename', 'unknown_file.pdf')}, Page: {metadata.get('page_number', 'N/A')}\nContent: {text_chunk}\n---"
            )

        context_str = "\n".join(context_parts_for_llm)

        direct_answer = llm_service.generate_direct_answer(request.query, context_str)

        identified_themes_list: Optional[List[Any]] = None
        if not request.doc_id:
            themes_output_str = llm_service.generate_themes(request.query, context_str)
            if themes_output_str and not themes_output_str.startswith("Error:"):
                identified_themes_list = llm_service.parse_llm_theme_response(
                    themes_output_str
                )
            else:
                logger.warning(
                    f"API: Theme generation from LLM failed or returned error: {themes_output_str}"
                )

        return QueryResponse(
            direct_answer=direct_answer,
            identified_themes=(
                identified_themes_list if identified_themes_list else None
            ),
            sources=source_chunks_for_response,
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(
            f"API: Unexpected error during query processing: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing your query: {str(e)}",
        )
