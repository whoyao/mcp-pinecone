import logging
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from .pinecone import PineconeClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pinecone-mcp")

pinecone_client = None
server = Server("pinecone-mcp")


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List all vectors as resources"""
    try:
        # List records from Pinecone
        records = await pinecone_client.list_records(limit=100)
        return [
            types.Resource(
                uri=f"pinecone://vectors/{record['id']}",
                name=record.get("metadata", {}).get("title", f"Vector {record['id']}"),
                description=record.get("metadata", {}).get("text", "")[:100] + "...",
                metadata=record.get("metadata", {}),
                mimeType="text/plain",
            )
            for record in records.get("records", [])
        ]
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        return []


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Read a specific vector's content"""
    if not str(uri).startswith("pinecone://vectors/"):
        raise ValueError(f"Unsupported URI scheme: {uri}")

    try:
        vector_id = str(uri).split("/")[-1]
        record = await pinecone_client.fetch_records([vector_id])

        if not record or "records" not in record or not record["records"]:
            raise ValueError(f"Vector not found: {vector_id}")

        vector_data = record["records"][0]

        # Format metadata and content as text
        output = []
        metadata = vector_data.get("metadata", {})

        if "title" in metadata:
            output.append(f"Title: {metadata['title']}")
        output.append(f"ID: {vector_id}")

        for key, value in metadata.items():
            if key != "title":
                output.append(f"{key}: {value}")

        output.append("")  # Empty line between metadata and content

        if "text" in metadata:
            output.append(metadata["text"])

        return "\n".join(output)

    except Exception as e:
        raise RuntimeError(f"Pinecone error: {str(e)}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools for Pinecone operations"""
    return [
        types.Tool(
            name="semantic-search",
            description="Semantic search across vectors",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Whether to use reranking",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="upsert-document",
            description="Add or update a document",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Document ID",
                    },
                    "text": {
                        "type": "string",
                        "description": "Document text content",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata",
                    },
                },
                "required": ["id", "text"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests"""

    if name == "semantic-search":
        query = arguments.get("query")
        top_k = arguments.get("top_k", 10)
        rerank = arguments.get("rerank", False)

        results = await pinecone_client.search_records(
            query=query, top_k=top_k, rerank={"enabled": rerank} if rerank else None
        )

        matches = results.get("matches", [])
        return [
            types.TextContent(
                type="text",
                text=f"Found {len(matches)} matches:\n"
                + "\n".join(
                    f"- {match.get('metadata', {}).get('title', f'Document {match['id']}')} "
                    f"(Score: {match['score']:.2f})"
                    for match in matches
                ),
            )
        ]

    elif name == "upsert-document":
        doc_id = arguments.get("id")
        text = arguments.get("text")
        metadata = arguments.get("metadata", {})

        record = {"_id": doc_id, "text": text, "metadata": metadata}

        await pinecone_client.upsert_records([record])

        return [
            types.TextContent(
                type="text", text=f"Successfully upserted document: {doc_id}"
            )
        ]

    else:
        raise ValueError(f"Unknown tool: {name}")

    # Notify clients that resources have changed
    await server.request_context.session.send_resource_list_changed()


async def main():
    """Run the server using stdin/stdout streams"""
    logger.info("Starting Pinecone MCP server")

    global pinecone_client
    pinecone_client = PineconeClient()

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pinecone-mcp",  # Hardcoded name instead of using metadata
                server_version="0.1.0",  # Hardcoded version instead of using metadata
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
