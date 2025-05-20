import os
import uuid
import logging
import aiofiles
import pdfplumber
from typing import List, Dict, Any, Tuple, Optional
from fastapi import UploadFile, HTTPException

from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import UPLOADED_PDFS_PATH, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP
from app.services.vector_store import add_documents_to_vector_store

logger = logging.getLogger(__name__)


document_metadata_store: Dict[str, Dict[str, Any]] = {}


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP
)


async def save_uploaded_file_locally(
    upload_file: UploadFile, destination_path: str
) -> None:
    try:
        async with aiofiles.open(destination_path, "wb") as out_file:
            while content := await upload_file.read(1024 * 1024):
                await out_file.write(content)
        logger.info(
            f"File '{upload_file.filename}' saved locally to '{destination_path}'."
        )
    except Exception as e:
        logger.error(
            f"Error saving file '{upload_file.filename}' to '{destination_path}': {e}",
            exc_info=True,
        )

        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise HTTPException(
            status_code=500, detail=f"Could not save uploaded file: {str(e)}"
        )


def extract_text_from_pdf_file(pdf_path: str) -> List[Dict[str, Any]]:

    pages_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                logger.warning(f"No pages found in PDF: {pdf_path}")

                return []

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append(
                        {
                            "page_content": text.strip(),
                            "metadata": {"page_number": i + 1},
                        }
                    )
                else:
                    logger.debug(
                        f"No text extracted from page {i+1} of PDF '{pdf_path}'. Might be image-based or empty."
                    )

        logger.info(f"Extracted text from {len(pages_data)} pages in PDF: {pdf_path}")
        return pages_data
    except Exception as e:
        logger.error(
            f"Failed to extract text from PDF '{pdf_path}': {e}", exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF file '{os.path.basename(pdf_path)}': Could not extract text.",
        )


def process_and_store_document(
    doc_id: str, original_filename: str, pdf_file_path: str
) -> None:
    global document_metadata_store
    try:
        document_metadata_store[doc_id]["status"] = "processing_text_extraction"
        logger.info(
            f"DocID {doc_id}: Starting text extraction for '{original_filename}'."
        )

        pages_with_content = extract_text_from_pdf_file(pdf_file_path)
        if not pages_with_content:
            document_metadata_store[doc_id]["status"] = "error_no_text_extracted"
            document_metadata_store[doc_id][
                "message"
            ] = "No text could be extracted from the PDF. It might be image-based or empty."
            logger.warning(
                f"DocID {doc_id}: No text extracted from '{original_filename}'."
            )

            return

        document_metadata_store[doc_id]["status"] = "processing_chunking"
        logger.info(
            f"DocID {doc_id}: Text extracted. Starting chunking for '{original_filename}'."
        )

        all_langchain_documents: List[LangchainDocument] = []
        for page_data in pages_with_content:
            page_text = page_data["page_content"]
            page_number = page_data["metadata"]["page_number"]

            text_chunks_on_page = text_splitter.split_text(page_text)

            for chunk_index, chunk in enumerate(text_chunks_on_page):
                chunk_metadata = {
                    "doc_id": doc_id,
                    "original_filename": original_filename,
                    "page_number": page_number,
                    "chunk_index_on_page": chunk_index,
                }
                all_langchain_documents.append(
                    LangchainDocument(page_content=chunk, metadata=chunk_metadata)
                )

        if not all_langchain_documents:
            document_metadata_store[doc_id]["status"] = "error_no_chunks_created"
            document_metadata_store[doc_id][
                "message"
            ] = "Text was extracted, but no processable chunks were created."
            logger.warning(
                f"DocID {doc_id}: No text chunks created for '{original_filename}' after splitting."
            )
            return

        logger.info(
            f"DocID {doc_id}: Created {len(all_langchain_documents)} chunks for '{original_filename}'."
        )
        document_metadata_store[doc_id]["status"] = "processing_embedding_storage"

        chroma_texts = [doc.page_content for doc in all_langchain_documents]
        chroma_metadatas = [doc.metadata for doc in all_langchain_documents]

        chroma_ids = [
            f"{doc_id}_pg{meta['page_number']}_chk{meta['chunk_index_on_page']}_{uuid.uuid4().hex[:6]}"
            for meta in chroma_metadatas
        ]

        add_documents_to_vector_store(
            ids=chroma_ids, texts=chroma_texts, metadatas=chroma_metadatas
        )

        document_metadata_store[doc_id]["status"] = "ready"
        document_metadata_store[doc_id][
            "message"
        ] = "Document processed and ready for querying."
        logger.info(
            f"DocID {doc_id}: Successfully processed and stored '{original_filename}'. Status: ready."
        )

    except HTTPException as http_exc:
        logger.error(
            f"DocID {doc_id}: HTTPException during processing of '{original_filename}': {http_exc.detail}",
            exc_info=True,
        )
        document_metadata_store[doc_id]["status"] = "error_processing"
        document_metadata_store[doc_id]["message"] = str(http_exc.detail)

        raise
    except Exception as e:
        logger.error(
            f"DocID {doc_id}: Unexpected error during processing of '{original_filename}': {e}",
            exc_info=True,
        )
        document_metadata_store[doc_id]["status"] = "error_processing_unexpected"
        document_metadata_store[doc_id][
            "message"
        ] = f"An unexpected error occurred: {str(e)}"

        raise HTTPException(
            status_code=500, detail=f"Unexpected error processing document: {str(e)}"
        )


def get_document_status(doc_id: str) -> Optional[Dict[str, Any]]:
    return document_metadata_store.get(doc_id)


def get_all_document_statuses() -> List[Dict[str, Any]]:
    return [{"doc_id": id, **data} for id, data in document_metadata_store.items()]


def remove_document_data(doc_id: str) -> bool:
    if doc_id not in document_metadata_store:
        logger.warning(
            f"Attempted to remove non-existent document metadata for doc_id: {doc_id}"
        )
        return False

    doc_meta = document_metadata_store.pop(doc_id)
    file_path = doc_meta.get("path")

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(
                f"Successfully deleted local file: {file_path} for doc_id: {doc_id}"
            )
        except Exception as e:
            logger.error(
                f"Error deleting local file {file_path} for doc_id {doc_id}: {e}",
                exc_info=True,
            )

    elif file_path:
        logger.warning(
            f"Local file path {file_path} for doc_id {doc_id} not found for deletion."
        )

    logger.info(f"Removed metadata for doc_id: {doc_id}")
    return True


def remove_all_documents_data() -> List[str]:
    doc_ids_removed = list(document_metadata_store.keys())
    for doc_id in list(document_metadata_store.keys()):
        remove_document_data(doc_id)
    logger.info(
        f"Removed all document metadata and local files. Count: {len(doc_ids_removed)}."
    )
    return doc_ids_removed
