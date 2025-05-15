import os
import re
import google.generativeai as genai
from typing import List, Dict, Any
from models import DocumentResponse


class ThemeIdentifier:

    def __init__(self):
        if not os.getenv("GOOGLE_API_KEY"):
            from dotenv import load_dotenv

            load_dotenv()
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

        self.model = genai.GenerativeModel("gemma-3-1b-it")

    def identify_themes(
        self, query: str, document_responses: List[DocumentResponse]
    ) -> Dict[str, Dict[str, Any]]:
        formatted_responses = ""
        for resp in document_responses:
            formatted_responses += f"Document ID: {resp.document_id}\n"
            formatted_responses += f"Answer: {resp.extracted_answer}\n"
            formatted_responses += f"Citation: {resp.citation}\n\n"
        prompt = f"""
        Analyze the following document responses to a query and identify common themes across them.
        Multiple themes are possible, and each theme should be supported by at least one document.
        
        QUERY: {query}
        
        DOCUMENT RESPONSES:
        {formatted_responses}
        
        For each identified theme, provide:
        1. A concise theme name
        2. A detailed description of the theme
        3. List of document IDs that support this theme
        
        Format your response as follows:
        THEME 1:
        NAME: [Theme name]
        DESCRIPTION: [Theme description]
        SUPPORTING_DOCUMENTS: [List of document IDs]
        
        THEME 2:
        NAME: [Theme name]
        DESCRIPTION: [Theme description]
        SUPPORTING_DOCUMENTS: [List of document IDs]
        
        And so on for each identified theme.
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            themes = self._parse_themes(response_text)
            return themes
        except Exception as e:
            print(f"Error identifying themes: {e}")
            return {
                "theme_1": {
                    "name": "General Information",
                    "description": "General information related to the query.",
                    "supporting_documents": [
                        resp.document_id for resp in document_responses
                    ],
                }
            }

    def _parse_themes(self, response_text: str) -> Dict[str, Dict[str, Any]]:
        themes = {}

        theme_sections = response_text.split("THEME ")[1:]

        for i, section in enumerate(theme_sections):
            theme_id = f"theme_{i+1}"

            name_match = re.search(
                r"NAME:\s*(.+?)(?=DESCRIPTION:|$)", section, re.DOTALL
            )
            desc_match = re.search(
                r"DESCRIPTION:\s*(.+?)(?=SUPPORTING_DOCUMENTS:|$)", section, re.DOTALL
            )
            docs_match = re.search(
                r"SUPPORTING_DOCUMENTS:\s*(.+?)(?=THEME|$)", section, re.DOTALL
            )

            if name_match and desc_match:
                name = name_match.group(1).strip()
                description = desc_match.group(1).strip()

                supporting_docs = []
                if docs_match:
                    docs_text = docs_match.group(1).strip()
                    doc_ids = re.findall(r"DOC\w+", docs_text)
                    supporting_docs = doc_ids if doc_ids else [docs_text]

                themes[theme_id] = {
                    "name": name,
                    "description": description,
                    "supporting_documents": supporting_docs,
                }

        if not themes:
            themes["theme_1"] = {
                "name": "General Information",
                "description": "General information related to the query.",
                "supporting_documents": [],
            }

        return themes
