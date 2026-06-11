# Single message is likely most reliable and generally better for this task
# No final reminders at the end since the user query is expected to be short
# If it is not short, it should go into the chat flow so we do not need to account for this.
KEYWORD_EXPANSION_PROMPT = """
Generate a set of keyword-only queries to help find relevant documents for the provided query. \
These queries will be passed to a bm25-based keyword search engine. \
Provide a single query per line (where each query consists of one or more keywords). \
The queries must be purely keywords and not contain any filler natural language. \
The each query should have as few keywords as necessary to represent the user's search intent. \
If there are no useful expansions, simply return the original query with no additional keyword queries. \
CRITICAL: Do not include any additional formatting, comments, or anything aside from the keyword queries.

The user query is:
{user_query}
""".strip()


QUERY_TYPE_PROMPT = """
Determine if the provided query is better suited for a keyword search or a semantic search.
Respond with "keyword" or "semantic" literally and nothing else.
Do not provide any additional text or reasoning to your response.

CRITICAL: It must only be 1 single word - EITHER "keyword" or "semantic".

The user query is:
{user_query}
""".strip()

REPHRASE_CONTEXT_PROMPT = """
# Additional Context
User information: {user_info}
User memories:
{memories}
""".strip()

SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT = """
You are a search query rewriter. Today is {current_date}.
Given a conversation history and a user query, rewrite the query into a standalone, self-contained search query.
The rephrased query should incorporate necessary context from the conversation history and be optimized for semantic search.
Respond with ONLY the rephrased query — no explanation, no quotes.
""".strip()

SEMANTIC_QUERY_REPHRASE_USER_PROMPT = """{additional_context}
Rephrase the following query into a standalone search query:
{user_query}""".strip()

KEYWORD_REPHRASE_SYSTEM_PROMPT = """
You are a keyword query generator. Today is {current_date}.
Given a conversation history and a user query, generate up to 3 keyword-only search queries.
Each query should consist of only keywords (no natural language filler words), be on a separate line, and be optimized for keyword/BM25-based search.
Respond with ONLY the keyword queries, one per line, no explanations.
""".strip()

KEYWORD_REPHRASE_USER_PROMPT = """{additional_context}
Generate keyword search queries for:
{user_query}""".strip()
