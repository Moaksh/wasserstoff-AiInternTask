import hashlib
import logging
import os
import pickle
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from chromadb.utils import embedding_functions
from models import Document


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vector_store")


class QueryCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.lock = threading.RLock()

    def get(self, query, k, filter_dict=None):
        with self.lock:

            key = f"{query}_{k}_{filter_dict}"
            key_hash = hashlib.md5(key.encode()).hexdigest()

            if key_hash in self.cache:

                self.access_times[key_hash] = time.time()
                logger.info(f"Query cache hit for: {query[:30]}...")
                return self.cache[key_hash]

        return None

    def set(self, query, k, filter_dict, results):
        with self.lock:

            key = f"{query}_{k}_{filter_dict}"
            key_hash = hashlib.md5(key.encode()).hexdigest()

            self.cache[key_hash] = results
            self.access_times[key_hash] = time.time()

            if len(self.cache) > self.max_size:
                lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[lru_key]
                del self.access_times[lru_key]


class VectorStore:

    def __init__(self, persist_directory: str, batch_size: int = 20):

        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)

        self.client.settings = {"anonymized_telemetry": False, "allow_reset": True}

        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 96,
                "hnsw:M": 16,
            },
        )

        self.batch_size = batch_size

        self.query_cache = QueryCache(max_size=200)

        self.batch_buffer = {"ids": [], "documents": [], "metadatas": []}
        self.batch_lock = threading.RLock()

    def add_document(self, document: Document) -> None:
        start_time = time.time()

        chunks = self._chunk_document(document)

        with self.batch_lock:
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document.id}_chunk_{i}"
                metadata = {
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunk_index": i,
                    "timestamp": time.time(),
                    "total_chunks": len(chunks),
                }

                self.batch_buffer["ids"].append(chunk_id)
                self.batch_buffer["documents"].append(chunk)
                self.batch_buffer["metadatas"].append(metadata)

                if len(self.batch_buffer["ids"]) >= self.batch_size:
                    self._process_batch()

            self._process_batch()

        end_time = time.time()
        logger.info(
            f"Document added with {len(chunks)} chunks in {end_time - start_time:.2f} seconds"
        )

    def _process_batch(self):
        if not self.batch_buffer["ids"]:
            return

        batch_size = len(self.batch_buffer["ids"])

        try:

            self.collection.add(
                ids=self.batch_buffer["ids"],
                documents=self.batch_buffer["documents"],
                metadatas=self.batch_buffer["metadatas"],
            )

            self.batch_buffer = {"ids": [], "documents": [], "metadatas": []}

            logger.info(f"Processed batch of {batch_size} chunks")
        except Exception as e:
            logger.error(f"Error processing batch: {e}")

            for i in range(len(self.batch_buffer["ids"])):
                try:
                    self.collection.add(
                        ids=[self.batch_buffer["ids"][i]],
                        documents=[self.batch_buffer["documents"][i]],
                        metadatas=[self.batch_buffer["metadatas"][i]],
                    )
                except Exception as inner_e:
                    logger.error(f"Error adding individual document: {inner_e}")

            self.batch_buffer = {"ids": [], "documents": [], "metadatas": []}

    def flush(self):
        with self.batch_lock:
            self._process_batch()

    def _chunk_document(
        self, document: Document, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        content = document.content
        chunks = []

        paragraphs = re.split(r"\n\s*\n", content)

        current_chunk = ""
        for paragraph in paragraphs:

            if not paragraph.strip():
                continue

            if len(current_chunk) + len(paragraph) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                if len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:] + "\n" + paragraph
                else:
                    current_chunk = paragraph
            else:

                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if not chunks:
            for i in range(0, len(content), chunk_size - chunk_overlap):
                chunk = content[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())

        logger.info(f"Document chunked into {len(chunks)} chunks")
        return chunks

    def query(
        self, query_text: str, doc_ids: Optional[List[str]] = None, limit: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        start_time = time.time()

        with self.batch_lock:
            self._process_batch()

        if self.collection.count() == 0:
            logger.warning("Vector store is empty - documents may still be processing")

            if doc_ids:
                return {
                    doc_id: [
                        {
                            "text": "Document processing started. Please try again in a moment.",
                            "metadata": {"document_id": doc_id},
                        }
                    ]
                    for doc_id in doc_ids
                }
            else:

                return {}

        filter_dict = None
        if doc_ids:

            existing_docs = set()
            try:

                all_ids = self.collection.get(include=[])["metadatas"]
                for metadata in all_ids:
                    if "document_id" in metadata:
                        existing_docs.add(metadata["document_id"])

                missing_docs = [
                    doc_id for doc_id in doc_ids if doc_id not in existing_docs
                ]

                if missing_docs:
                    logger.warning(
                        f"Some requested documents are not in vector store yet: {missing_docs}"
                    )

                    results = {
                        doc_id: [
                            {
                                "text": "Document processing started. Please try again in a moment.",
                                "metadata": {"document_id": doc_id},
                            }
                        ]
                        for doc_id in missing_docs
                    }

                    existing_doc_ids = [
                        doc_id for doc_id in doc_ids if doc_id in existing_docs
                    ]
                    if not existing_doc_ids:

                        return results

                    doc_ids = existing_doc_ids
            except Exception as e:
                logger.error(f"Error checking document existence: {e}")

            filter_dict = {"document_id": {"$in": doc_ids}}

        cached_results = self.query_cache.get(query_text, limit, filter_dict)
        if cached_results is not None:
            return cached_results

        missing_doc_results = {}
        if "results" in locals():
            missing_doc_results = results.copy()

        results = self.collection.query(
            query_texts=[query_text],
            n_results=limit,
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )

        grouped_results = {}
        document_scores = {}

        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                document_id = metadata["document_id"]
                distance = results["distances"][0][i] if "distances" in results else 1.0

                if document_id not in grouped_results:
                    grouped_results[document_id] = []
                    document_scores[document_id] = []

                grouped_results[document_id].append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": metadata,
                        "relevance_score": 1.0 - distance,
                    }
                )
                document_scores[document_id].append(1.0 - distance)

        if document_scores:
            avg_scores = {
                doc_id: sum(scores) / len(scores)
                for doc_id, scores in document_scores.items()
            }
            sorted_results = {
                k: grouped_results[k]
                for k in sorted(
                    grouped_results.keys(), key=lambda x: avg_scores[x], reverse=True
                )
            }
        else:
            sorted_results = {}

        if "missing_doc_results" in locals() and missing_doc_results:
            logger.info(f"Merging results with processing status for missing documents")
            sorted_results.update(missing_doc_results)

        self.query_cache.set(query_text, limit, filter_dict, sorted_results)

        end_time = time.time()
        logger.info(
            f"Query executed in {end_time - start_time:.4f} seconds, found {len(sorted_results)} results"
        )

        return sorted_results

    def get_all_documents(self) -> List[Dict[str, Any]]:

        all_items = self.collection.get()

        doc_ids = set()
        for metadata in all_items["metadatas"]:
            doc_ids.add(metadata["document_id"])

        documents = []
        for doc_id in doc_ids:

            for i, metadata in enumerate(all_items["metadatas"]):
                if metadata["document_id"] == doc_id:
                    documents.append({"id": doc_id, "filename": metadata["filename"]})
                    break

        return documents

    def clear(self) -> None:

        try:

            all_items = self.collection.get()

            if all_items and all_items["ids"]:
                self.collection.delete(ids=all_items["ids"])

            return True
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        try:

            all_items = self.collection.get()

            doc_chunk_ids = []
            for i, metadata in enumerate(all_items["metadatas"]):
                if metadata["document_id"] == doc_id:
                    doc_chunk_ids.append(all_items["ids"][i])

            if doc_chunk_ids:
                self.collection.delete(ids=doc_chunk_ids)
                return True
            return False
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False
