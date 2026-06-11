"""Prompts for user memory management."""

FULL_MEMORY_UPDATE_PROMPT = """
You are managing a user's personal memory list.
Given a new memory and existing memories, determine whether to ADD the new memory or UPDATE an existing one.
{user_basic_information}

Recent chat history:
{chat_history}

Existing memories:
{existing_memories}

New memory to process:
{new_memory}

Respond with a JSON object:
- To add a new memory: {{"operation": "add", "memory_text": "<final memory text>"}}
- To update an existing memory: {{"operation": "update", "memory_id": <1-indexed integer>, "memory_text": "<updated text>"}}

Consider updating when the new memory significantly overlaps or supersedes an existing one.
Respond with ONLY the JSON object, no explanation.
""".strip()
