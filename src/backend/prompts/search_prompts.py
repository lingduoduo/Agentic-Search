"""Prompts for document search, section selection, and context expansion."""

MAX_CHUNKS_FOR_RELEVANCE = 10

TRY_TO_FILL_TO_MAX_INSTRUCTIONS = "Try to fill up to the maximum number of sections if there are potentially relevant sections."

DOCUMENT_SELECTION_PROMPT = """
You are selecting the most relevant document sections to answer a user query.
Select up to {max_sections} sections from the candidates below.
{extra_instructions}
Return ONLY a JSON array of section IDs (integers), e.g.: [0, 3, 7]
If no sections are relevant, return an empty array: []

User query: {user_query}

Document sections:
{formatted_doc_sections}
""".strip()

DOCUMENT_CONTEXT_SELECTION_PROMPT = """
You are evaluating whether a document section is relevant to a user query and how much context to include.

Document title: {document_title}
User query: {user_query}

Main section:
{main_section}

Section above (if any):
{section_above}

Section below (if any):
{section_below}

Rate relevance and context expansion using one of these situations:
0 - Not relevant: The section does not address the user query
1 - Main section only: The section is relevant as-is, no adjacent context needed
2 - Include adjacent sections: The section is relevant but needs adjacent context for completeness
3 - Full document: The full document context is needed to properly answer the query

Respond with ONLY the situation number (0, 1, 2, or 3).
""".strip()
