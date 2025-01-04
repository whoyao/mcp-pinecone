import logging
from enum import Enum
import mcp.types as types
from mcp.server import Server
from .pinecone import PineconeClient


logger = logging.getLogger("pinecone-mcp")


class PromptName(str, Enum):
    PINECONE_QUERY = "pinecone-query"


ServerPrompts = [
    types.Prompt(
        name=PromptName.PINECONE_QUERY,
        description="Search Pinecone index and construct an answer based on relevant pinecone documents",
        arguments=[
            types.PromptArgument(
                name="query",
                description="The question to answer, or the context to search for",
                required=True,
            )
        ],
    )
]


def register_prompts(server: Server, pinecone_client: PineconeClient):
    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return ServerPrompts

    @server.get_prompt()
    async def handle_get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> types.GetPromptResult:
        try:
            if name == PromptName.PINECONE_QUERY:
                return pinecone_query(arguments, pinecone_client)

        except Exception as e:
            logger.error(f"Error calling prompt {name}: {e}")
            raise


def pinecone_query(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Search Pinecone index and construct an answer based on relevant pinecone documents
    """
    query = arguments.get("query")
    if not query:
        raise ValueError("Query required")

    return types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="First use pinecone-stats to get a list of namespaces that might contain relevant documents. Ignore if a namespace is specified in the query",
                ),
            ),
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Do a semantic search for the query: {query} with the chosen namespace",
                ),
            ),
        ]
    )


__all__ = [
    "register_prompts",
]
