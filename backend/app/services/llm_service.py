import logging
import re
from typing import List, Optional, Tuple

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from app.core.config import (
    GOOGLE_API_KEY, 
    EMBEDDING_MODEL_NAME, 
    GENERATION_MODEL_NAME, 
    LLM_TEMPERATURE
)
from app.models.pydantic_models import IdentifiedTheme

logger = logging.getLogger(__name__) 
theme_parser_logger = logging.getLogger("theme_parser") 

try:
    google_embeddings_instance = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME, 
        google_api_key=GOOGLE_API_KEY
    )
    llm_instance = ChatGoogleGenerativeAI(
        model=GENERATION_MODEL_NAME, 
        google_api_key=GOOGLE_API_KEY, 
        temperature=LLM_TEMPERATURE
    )
    logger.info(f"LLM Service: Initialized Google Embeddings with model: {EMBEDDING_MODEL_NAME}")
    logger.info(f"LLM Service: Initialized ChatGoogleGenerativeAI with model: {GENERATION_MODEL_NAME}, temp: {LLM_TEMPERATURE}")
except Exception as e:
    logger.critical(f"LLM Service: Error initializing Google GenAI components: {e}", exc_info=True)
    google_embeddings_instance = None
    llm_instance = None

def get_llm():
    if not llm_instance:
        logger.error("LLM instance is not available.")

        raise RuntimeError("LLM service not initialized properly.")
    return llm_instance

def get_embeddings():
    if not google_embeddings_instance:
        logger.error("Embeddings instance is not available.")
        raise RuntimeError("Embeddings service not initialized properly.")
    return google_embeddings_instance

def generate_direct_answer(query: str, context_str: str) -> str:
    llm = get_llm()
    prompt = f"""
    You are an AI assistant. Based ONLY on the following document excerpts, provide a direct and concise answer to the user's query.
    If the information is not in the excerpts, clearly state that. Do not make up information.

    User Query: "{query}"

    Excerpts:
    ---
    {context_str}
    ---

    Direct Answer:
    """
    logger.info("LLM Service: Sending request to LLM for direct answer.")
    logger.debug(f"LLM Service: Prompt (Direct Answer):\n'''{prompt[:600]}...'''") 
    
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        logger.info(f"LLM Service: Response (Direct Answer received): '''{answer[:200]}...'''")
        return answer
    except Exception as e:
        logger.error(f"LLM Service: Error during direct answer generation: {e}", exc_info=True)
        return "Error: Could not generate a direct answer due to an internal issue."


def generate_themes(query: str, context_str: str) -> str:
    llm = get_llm()
    prompt = f"""
    You are an AI assistant specialized in identifying themes from text. Based on the provided document excerpts, perform the following tasks meticulously:
    1.  Identify the main themes (aim for 2-4 distinct themes if present) that emerge from these excerpts in relation to the User Query.
    2.  For each identified theme, you MUST structure the information as follows, using these exact phrases and newlines:
        Theme [Number]: [Descriptive Title of the Theme]
        Supporting Documents: [doc_id_x, doc_id_y] (List unique Document IDs from the excerpts' metadata. Use the exact Document IDs as provided in the context. If no documents support a theme from the provided excerpts, use an empty list: [])
        Summary: [A brief summary, 2-3 sentences, explaining what the documents say about this theme in relation to the User Query.]
    3.  Ensure each complete theme block (Theme, Supporting Documents, Summary) is clearly distinct.
    4.  If no clear themes are found related to the query, simply output "No specific themes identified from the provided excerpts related to the query."

    User Query: "{query}"

    Excerpts from Documents (Pay close attention to 'Document ID' in each excerpt for citation):
    ---
    {context_str}
    ---

    Your response should ONLY contain the theme information, starting with "Identified Themes:" or the "No specific themes..." message.
    Example of desired theme output:
    Identified Themes:
    Theme 1: [Title of Theme 1]
    Supporting Documents: [doc_id_abc, doc_id_xyz]
    Summary: [Brief summary of Theme 1]

    Theme 2: [Title of Theme 2]
    Supporting Documents: [doc_id_123]
    Summary: [Brief summary of Theme 2]

    Begin your response now:
    """
    logger.info("LLM Service: Sending request to LLM for theme identification.")
    theme_parser_logger.debug(f"LLM Service: Prompt (Themes):\n'''{prompt[:800]}...'''")
    
    try:
        response = llm.invoke(prompt)
        themes_output = response.content.strip()
        theme_parser_logger.debug(f"LLM Service: Response (Themes raw received):\n'''{themes_output}'''")
        return themes_output
    except Exception as e:
        logger.error(f"LLM Service: Error during theme generation: {e}", exc_info=True)
        return "Error: Could not generate themes due to an internal issue."


def parse_llm_theme_response(llm_theme_output: str) -> List[IdentifiedTheme]:
    themes: List[IdentifiedTheme] = []
    llm_output_cleaned = llm_theme_output.strip()

    theme_parser_logger.debug(f"\n--- Theme Parser: Raw LLM Output for Theme Parsing ---\n{llm_output_cleaned}\n--------------------------------------------------\n")

    try:
        themes_block_text = llm_output_cleaned
        if llm_output_cleaned.lower().startswith("identified themes:"):
            themes_block_text = llm_output_cleaned[len("identified themes:"):].strip()
            theme_parser_logger.debug(f"Theme Parser: Stripped 'Identified Themes:' header. Remaining block:\n'''{themes_block_text[:300]}...'''")
        
        if not themes_block_text or "no specific themes identified" in themes_block_text.lower():
            theme_parser_logger.info("Theme Parser: No themes indicated by LLM or themes block empty.")
            return themes

        potential_theme_chunks = [chunk.strip() for chunk in re.split(r"(?=Theme\s+\d+\s*:)", themes_block_text) if chunk.strip()]
        theme_parser_logger.debug(f"Theme Parser: Found {len(potential_theme_chunks)} potential theme chunks.")

        if not potential_theme_chunks and themes_block_text:
             theme_parser_logger.warning(f"Theme Parser: Could not split themes block. Content:\n'''{themes_block_text}'''")

        for i, theme_chunk_text in enumerate(potential_theme_chunks):
            theme_parser_logger.debug(f"\nTheme Parser: Processing Chunk {i+1}:\n'''{theme_chunk_text}'''")
            
            current_title = "Untitled Theme"
            doc_ids = []
            summary = "No summary provided."

            title_match = re.match(r"Theme\s+\d+\s*:\s*(.*)", theme_chunk_text, re.IGNORECASE)
            if title_match:
                current_title = title_match.group(1).split('\n')[0].strip()
                remaining_chunk_text = theme_chunk_text[title_match.end():].strip()
                theme_parser_logger.debug(f"  Title: '{current_title}'")
            else:
                theme_parser_logger.warning(f"  Title pattern not matched. Chunk: '{theme_chunk_text[:100]}...'")
                current_title = theme_chunk_text.split('\n')[0].strip() if theme_chunk_text else "Untitled Theme"
                remaining_chunk_text = '\n'.join(theme_chunk_text.split('\n')[1:]).strip() if '\n' in theme_chunk_text else ""

            docs_match = re.search(r"Supporting Documents\s*:\s*\[(.*?)\]", remaining_chunk_text, re.IGNORECASE | re.DOTALL)
            if docs_match:
                doc_ids_str = docs_match.group(1).strip()
                doc_ids = [doc.strip() for doc in doc_ids_str.split(',') if doc.strip()]
                theme_parser_logger.debug(f"  Docs: {doc_ids} (Raw: '[{doc_ids_str}]')")
            else:
                theme_parser_logger.debug(f"  'Supporting Documents:' not found.")
            
            summary_match = re.search(r"Summary\s*:\s*(.*)", remaining_chunk_text, re.IGNORECASE | re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1).strip()
                next_theme_start = re.search(r"\n\s*Theme\s+\d+\s*:", summary_text)
                if next_theme_start:
                    summary = summary_text[:next_theme_start.start()].strip()
                else:
                    summary = summary_text
                theme_parser_logger.debug(f"  Summary: '''{summary[:100]}...'''")
            else:
                theme_parser_logger.debug(f"  'Summary:' not found.")

            if current_title != "Untitled Theme" or doc_ids or (summary != "No summary provided." and summary.strip()):
                themes.append(IdentifiedTheme(theme_title=current_title, supporting_doc_ids=doc_ids, summary=summary))
                logger.info(f"LLM Service: Successfully Parsed and Added Theme: Title='{current_title}'")
            else:
                theme_parser_logger.warning(f"  Skipping chunk for '{current_title}', insufficient data.")
    
    except Exception as e:
        logger.error(f"LLM Service: CRITICAL ERROR in parse_llm_theme_response: {e}", exc_info=True)
    
    logger.info(f"LLM Service: parse_llm_theme_response finished. Total Themes Found: {len(themes)}")
    return themes

