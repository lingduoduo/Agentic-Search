"""Chat prompt string constants."""

IMAGE_GEN_REMINDER = "Please describe or display the generated image in your response."
OPEN_URL_REMINDER = (
    "You can open URLs from the search results above to get more information."
)
CITATION_REMINDER = "Cite your sources using [1], [2] etc."
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
FILE_REMINDER = "You have access to files. Reference them by their file ID."
LAST_CYCLE_CITATION_REMINDER = "Make sure to cite sources in your final answer."
REQUIRE_CITATION_GUIDANCE = "You MUST cite all claims with [n] markers."
ADDITIONAL_CONTEXT_PROMPT = "Here is additional context:"
TOOL_CALL_RESPONSE_CROSS_MESSAGE = "Tool response:"
CODE_BLOCK_MARKDOWN = "```"
IMAGE_DROP_REMINDER = "Do not generate image links in markdown format."

# Tool prompt stubs (full text provided by real implementations)
TOOL_CALL_FAILURE_PROMPT: str = (
    "Tool call failed. Please try again or use a different approach."
)
TOOL_CALL_RESPONSE_CROSS_MESSAGE: str = "Tool response:"
ADDITIONAL_CONTEXT_PROMPT: str = "Additional context:\n{additional_context}"
