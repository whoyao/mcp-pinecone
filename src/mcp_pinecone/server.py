import logging
from typing import Union
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from .pinecone import PineconeClient
from .tools import register_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pinecone-mcp")

pinecone_client = None
server = Server("pinecone-mcp")


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    try:
        if pinecone_client is None:
            logger.error("Pinecone client is not initialized")
            return []
        records = pinecone_client.list_records()

        resources = []
        for record in records.get("vectors", []):
            # If metadata is None, use empty dict
            metadata = record.get("metadata") or {}
            description = (
                metadata.get("text", "")[:100] + "..." if metadata.get("text") else ""
            )
            resources.append(
                types.Resource(
                    uri=f"pinecone://vectors/{record['id']}",
                    name=metadata.get("title", f"Vector {record['id']}"),
                    description=description,
                    metadata=metadata,
                    mimeType=metadata.get("content_type", "text/plain"),
                )
            )
        return resources
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        return []


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> Union[str, bytes]:
    if not str(uri).startswith("pinecone://vectors/"):
        raise ValueError(f"Unsupported URI scheme: {uri}")

    try:
        vector_id = str(uri).split("/")[-1]
        record = pinecone_client.fetch_records([vector_id])

        if not record or "records" not in record or not record["records"]:
            raise ValueError(f"Vector not found: {vector_id}")

        vector_data = record["records"][0]
        metadata = vector_data.get("metadata", {})
        content_type = metadata.get("content_type", "text/plain")

        if content_type.startswith("text/"):
            return format_text_content(vector_data)
        else:
            return format_binary_content(vector_data)
    except Exception as e:
        raise RuntimeError(f"Pinecone error: {str(e)}")


def format_text_content(vector_data: dict) -> str:
    metadata = vector_data.get("metadata", {})
    output = []

    if "title" in metadata:
        output.append(f"Title: {metadata['title']}")
    output.append(f"ID: {vector_data.get('id')}")

    for key, value in metadata.items():
        if key not in ["title", "text", "content_type"]:
            output.append(f"{key}: {value}")

    output.append("")

    if "text" in metadata:
        output.append(metadata["text"])

    return "\n".join(output)


def format_binary_content(vector_data: dict) -> bytes:
    content = vector_data.get("metadata", {}).get("content", b"")
    if isinstance(content, str):
        content = content.encode("utf-8")
    return content


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name == "pinecone-query":
        query = arguments.get("query")
        if not query:
            raise ValueError("Query required")

        # The semantic search already includes the full text in the metadata
        # It's formatted in search_results[0].text which contains:
        # - Document contents
        # - Similarity scores
        # - Separators between documents

        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"The following documents match the search query and will provide context: {search_results[0].text}",
                    ),
                ),
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Answer the question: {query}",
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown prompt: {name}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="pinecone-query",
            description="Search knowledge base and construct an answer based on relevant pinecone documents",
            arguments=[
                types.PromptArgument(
                    name="query",
                    description="The question to answer",
                    required=True,
                )
            ],
        )
    ]


async def main():
    logger.info("Starting Pinecone MCP server")

    global pinecone_client
    pinecone_client = PineconeClient()

    # Register tools
    register_tools(server, pinecone_client)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pinecone-mcp",
                server_version="0.1.7",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(resources_changed=True),
                    experimental_capabilities={},
                ),
            ),
        )
