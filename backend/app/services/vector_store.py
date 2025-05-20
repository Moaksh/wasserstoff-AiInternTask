import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional

from app.core.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    GOOGLE_API_KEY,
    EMBEDDING_MODEL_NAME,
)
from app.services.llm_service import get_embeddings

logger = logging.getLogger(__name__)


_chroma_client: Optional[chromadb.Client] = None
_collection: Optional[chromadb.Collection] = None


def get_chroma_client() -> chromadb.Client:
    global _chroma_client
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            logger.info(
                f"ChromaDB persistent client initialized at path: {CHROMA_DB_PATH}"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize persistent ChromaDB client: {e}. Falling back to in-memory.",
                exc_info=True,
            )
            _chroma_client = chromadb.Client()
    return _chroma_client


def get_vector_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_chroma_client()
        try:

            google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=GOOGLE_API_KEY, model_name=EMBEDDING_MODEL_NAME
            )
            _collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME, embedding_function=google_ef
            )
            logger.info(
                f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' obtained/created successfully using model '{EMBEDDING_MODEL_NAME}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to get/create ChromaDB collection '{CHROMA_COLLECTION_NAME}': {e}",
                exc_info=True,
            )

            raise RuntimeError(f"Could not initialize ChromaDB collection: {e}")
    return _collection


def add_documents_to_vector_store(
    ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]
):
    collection = get_vector_collection()
    try:
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logger.info(
            f"Added {len(texts)} chunks to ChromaDB collection '{CHROMA_COLLECTION_NAME}'."
        )
    except Exception as e:
        logger.error(f"Failed to add documents to ChromaDB: {e}", exc_info=True)
        raise


def query_vector_store(
    query_texts: List[str],
    n_results: int,
    where_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    collection = get_vector_collection()
    try:
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where_filter,
            include=["metadatas", "documents"],
        )
        logger.info(
            f"ChromaDB query returned {len(results.get('ids', [[]])[0])} results for query: '{query_texts[0][:50]}...'."
        )
        return results
    except Exception as e:
        logger.error(f"Failed to query ChromaDB: {e}", exc_info=True)
        raise


def delete_from_vector_store_by_ids(ids: List[str]):
    if not ids:
        logger.info("No IDs provided for deletion from vector store.")
        return
    collection = get_vector_collection()
    try:
        collection.delete(ids=ids)
        logger.info(
            f"Deleted {len(ids)} items from ChromaDB collection '{CHROMA_COLLECTION_NAME}'."
        )
    except Exception as e:
        logger.error(f"Failed to delete items from ChromaDB by IDs: {e}", exc_info=True)
        raise


def delete_from_vector_store_by_metadata(where_filter: Dict[str, Any]):
    if not where_filter:
        logger.warning("No metadata filter provided for deletion from vector store.")
        return
    collection = get_vector_collection()
    try:

        results_to_delete = collection.get(where=where_filter, include=[])
        ids_to_delete = results_to_delete.get("ids", [])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(
                f"Deleted {len(ids_to_delete)} items from ChromaDB based on metadata filter: {where_filter}."
            )
        else:
            logger.info(
                f"No items found in ChromaDB matching metadata filter for deletion: {where_filter}."
            )

    except Exception as e:
        logger.error(
            f"Failed to delete items from ChromaDB by metadata: {e}", exc_info=True
        )
        raise


def clear_vector_collection():
    global _collection
    client = get_chroma_client()
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        logger.info(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' deleted.")
        _collection = None
        get_vector_collection()
        logger.info(
            f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' recreated after clearing."
        )
    except Exception as e:
        logger.error(
            f"Failed to clear and recreate ChromaDB collection '{CHROMA_COLLECTION_NAME}': {e}",
            exc_info=True,
        )
        raise
