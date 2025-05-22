from fastapi import FastAPI
import logging


from app.core.config import APP_TITLE, APP_DESCRIPTION
from app.core.logging_config import setup_logging
from app.api import endpoints as api_endpoints
from app.services import vector_store
from app.services import llm_service


setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION)


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing resources...")
    try:

        vector_store.get_vector_collection()
        logger.info("ChromaDB connection checked/initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB on startup: {e}", exc_info=True)

    try:

        if not llm_service.llm_instance or not llm_service.google_embeddings_instance:
            logger.warning(
                "LLM or Embeddings not initialized on startup, llm_service might have failed."
            )

        else:
            logger.info("LLM and Embeddings services confirmed initialized.")
    except Exception as e:
        logger.critical(
            f"Failed during LLM/Embedding service check on startup: {e}", exc_info=True
        )

    logger.info("Application startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown.")


app.include_router(api_endpoints.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def read_root():
    logger.info("Root endpoint accessed.")
    return {
        "message": f"Welcome to the {APP_TITLE}",
        "docs_url": "/docs",
    }
