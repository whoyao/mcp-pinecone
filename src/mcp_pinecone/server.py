import logging
from typing import Union, Sequence
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from .pinecone import PineconeClient
from .utils import MCPToolError

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


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="semantic-search",
            description="Search pineconeknowledge base",
            category="search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                    "category": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string", "format": "date"},
                            "end": {"type": "string", "format": "date"},
                        },
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="read-document",
            description="Read a document from the pinecone knowledge base",
            category="read",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                },
                "required": ["document_id"],
            },
        ),
        types.Tool(
            name="upsert-document",
            description="Add or update content in the pinecone knowledge base",
            category="mutation",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["id", "text"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> Sequence[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
    try:
        if name == "semantic-search":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 10)
            filters = arguments.get("filters")

            results = pinecone_client.search_records(
                query=query, top_k=top_k, filter=filters, include_metadata=True
            )

            matches = results.get("matches", [])

            # Format results with rich context
            formatted_text = "Retrieved Contexts:\n\n"
            for i, match in enumerate(matches, 1):
                metadata = match.get("metadata", {})
                formatted_text += f"[Result {i} - Similarity: {match['score']:.3f}]\n"
                formatted_text += f"Document ID: {match['id']}\n"
                formatted_text += f"{metadata.get('text', '').strip()}\n"
                formatted_text += "-" * 40 + "\n\n"

            return [types.TextContent(type="text", text=formatted_text)]

        elif name == "read-document":
            document_id = arguments.get("document_id")
            if not document_id:
                raise ValueError("document_id is required")

            # Fetch the record using your existing fetch_records method
            record = pinecone_client.fetch_records([document_id])

            # Get the vector data for this document
            vector = record.vectors.get(document_id)
            if not vector:
                raise ValueError(f"Document {document_id} not found")

            # Get metadata from the vector
            metadata = vector.metadata if hasattr(vector, "metadata") else {}

            # Format the document content
            formatted_content = []
            formatted_content.append(f"Document ID: {document_id}")
            formatted_content.append("")  # Empty line for spacing

            if metadata:
                formatted_content.append("Metadata:")
                for key, value in metadata.items():
                    formatted_content.append(f"{key}: {value}")

            return [types.TextContent(type="text", text="\n".join(formatted_content))]

        elif name == "upsert-document":
            doc_id = arguments.get("id")
            text = arguments.get("text")
            metadata = arguments.get("metadata", {})

            # Use text directly - Pinecone will generate the embedding
            record = {
                "id": doc_id,
                "text": text,  # Pinecone will generate embedding from this
                "metadata": {**metadata},
            }

            pinecone_client.upsert_records([record])

            return [
                types.TextContent(
                    type="text", text=f"Successfully upserted document: {doc_id}"
                )
            ]
        else:
            raise MCPToolError(types.METHOD_NOT_FOUND, f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error in upsert: {str(e)}", exc_info=True)
        raise


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name == "brain-query":
        query = arguments.get("query")
        if not query:
            raise ValueError("Query required")

        # Get semantic search results
        search_results = await handle_call_tool(
            "semantic-search", {"query": query, "top_k": 3}
        )

        # The semantic search already includes the full text in the metadata
        # It's formatted in search_results[0].text which contains:
        # - Document contents
        # - Similarity scores
        # - Separators between documents

        prompt = f"""Human: You are an intelligent assistant tasked with answering questions based on the provided context. 
Your responses should be:
1. Accurate and directly based on the provided context
2. Well-structured and clear
3. Include relevant quotes when appropriate
4. Acknowledge any limitations in the available information

Retrieved Context:
{search_results[0].text}

Question: {query}

Please provide a comprehensive response that:
1. Directly answers the question
2. Cites specific evidence from the context
3. Notes any relevant connections between different parts of the context
4. Acknowledges any areas where the context may be incomplete"""

        return types.GetPromptResult(prompt=prompt)

    raise ValueError(f"Unknown prompt: {name}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="brain-query",
            description="Search knowledge base and construct an answer based on relevant pinecone documents",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )
    ]


async def main():
    logger.info("Starting Pinecone MCP server")

    global pinecone_client
    pinecone_client = PineconeClient()

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pinecone-mcp",
                server_version="0.1.2",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(resources_changed=True),
                    experimental_capabilities={},
                ),
            ),
        )
