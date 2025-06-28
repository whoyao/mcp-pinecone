import logging
import asyncio
import json
from typing import Union, AsyncGenerator
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from .pinecone import PineconeClient
from .tools import register_tools
from .prompts import register_prompts
import importlib.metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pinecone-mcp-sse")

pinecone_client = None
server = Server("pinecone-mcp-sse")


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


async def main():
    logger.info("Starting Pinecone MCP server with SSE support")

    global pinecone_client
    pinecone_client = PineconeClient()

    # Register tools and prompts with SSE support
    register_tools_sse(server, pinecone_client)
    register_prompts(server, pinecone_client)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pinecone-mcp-sse",
                server_version=importlib.metadata.version("mcp-pinecone"),
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(resources_changed=True),
                    experimental_capabilities={
                        "sse_support": True,
                        "streaming_tools": True
                    },
                ),
            ),
        )


def register_tools_sse(server: Server, pinecone_client: PineconeClient):
    """Register tools with SSE support for streaming responses"""
    
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="semantic-search-stream",
                description="Search pinecone for documents with streaming results",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace to search in",
                        },
                        "category": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string", "format": "date"},
                                "end": {"type": "string", "format": "date"},
                            },
                        },
                        "stream": {
                            "type": "boolean",
                            "description": "Enable streaming response",
                            "default": True
                        }
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="pinecone-stats-stream",
                description="Get stats about the Pinecone index with streaming response",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "stream": {
                            "type": "boolean",
                            "description": "Enable streaming response",
                            "default": True
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="process-document-stream",
                description="Process a document with streaming progress updates",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object"},
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace to store the document in",
                        },
                        "stream": {
                            "type": "boolean",
                            "description": "Enable streaming response",
                            "default": True
                        }
                    },
                    "required": ["document_id", "text", "metadata"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> AsyncGenerator[types.TextContent, None]:
        try:
            if name == "semantic-search-stream":
                async for content in semantic_search_stream(arguments, pinecone_client):
                    yield content
            elif name == "pinecone-stats-stream":
                async for content in pinecone_stats_stream(pinecone_client):
                    yield content
            elif name == "process-document-stream":
                async for content in process_document_stream(arguments, pinecone_client):
                    yield content
            else:
                # Fallback to non-streaming tools
                from .tools import (
                    semantic_search, pinecone_stats, process_document,
                    ToolName
                )
                
                if name == ToolName.SEMANTIC_SEARCH:
                    result = semantic_search(arguments, pinecone_client)
                    for content in result:
                        yield content
                elif name == ToolName.PINECONE_STATS:
                    result = pinecone_stats(pinecone_client)
                    for content in result:
                        yield content
                elif name == ToolName.PROCESS_DOCUMENT:
                    result = process_document(arguments, pinecone_client)
                    for content in result:
                        yield content

        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            yield types.TextContent(type="text", text=f"Error: {str(e)}")


async def semantic_search_stream(
    arguments: dict | None, pinecone_client: PineconeClient
) -> AsyncGenerator[types.TextContent, None]:
    """Streaming semantic search with progress updates"""
    try:
        query = arguments.get("query")
        top_k = arguments.get("top_k", 10)
        filters = arguments.get("filters", {})
        namespace = arguments.get("namespace")
        stream = arguments.get("stream", True)

        if not stream:
            # Fallback to non-streaming version
            from .tools import semantic_search
            result = semantic_search(arguments, pinecone_client)
            for content in result:
                yield content
            return

        # Send initial status
        yield types.TextContent(
            type="text", 
            text="🔍 Starting semantic search...\n"
        )
        await asyncio.sleep(0.1)  # Small delay for streaming effect

        # Generate embeddings
        yield types.TextContent(
            type="text", 
            text="🧠 Generating embeddings for query...\n"
        )
        await asyncio.sleep(0.1)

        # Perform search
        yield types.TextContent(
            type="text", 
            text="📡 Searching Pinecone index...\n"
        )
        await asyncio.sleep(0.1)

        results = pinecone_client.search_records(
            query=query,
            top_k=top_k,
            filter=filters,
            include_metadata=True,
            namespace=namespace,
        )

        # Pinecone query returns results with 'matches' key
        matches = results.get("matches", [])
        
        yield types.TextContent(
            type="text", 
            text=f"✅ Found {len(matches)} results\n\n"
        )

        # Stream results one by one
        for i, match in enumerate(matches, 1):
            metadata = match.get("metadata", {})
            score = match.get("score", 0)
            doc_id = match.get("id", "")
            
            result_text = f"📄 Result {i}:\n"
            result_text += f"   Similarity: {score:.3f}\n"
            result_text += f"   Document ID: {doc_id}\n"
            result_text += f"   Content: {metadata.get('text', '')[:100]}...\n"
            result_text += "-" * 40 + "\n"
            
            yield types.TextContent(type="text", text=result_text)
            await asyncio.sleep(0.05)  # Small delay between results

        yield types.TextContent(
            type="text", 
            text="🎉 Search completed!\n"
        )

    except Exception as e:
        yield types.TextContent(type="text", text=f"❌ Search failed: {str(e)}\n")


async def pinecone_stats_stream(
    pinecone_client: PineconeClient
) -> AsyncGenerator[types.TextContent, None]:
    """Streaming Pinecone stats with detailed breakdown"""
    try:
        yield types.TextContent(
            type="text", 
            text="📊 Retrieving Pinecone index statistics...\n"
        )
        await asyncio.sleep(0.1)

        stats = pinecone_client.stats()
        
        yield types.TextContent(
            type="text", 
            text="📈 Index Statistics:\n"
        )
        await asyncio.sleep(0.1)

        # Stream each stat
        total_vectors = stats.get('total_vector_count', 0)
        yield types.TextContent(
            type="text", 
            text=f"   Total vectors: {total_vectors}\n"
        )
        await asyncio.sleep(0.05)

        dimension = stats.get('dimension', 0)
        yield types.TextContent(
            type="text", 
            text=f"   Dimension: {dimension}\n"
        )
        await asyncio.sleep(0.05)

        index_fullness = stats.get('index_fullness', 0)
        yield types.TextContent(
            type="text", 
            text=f"   Index fullness: {index_fullness:.2%}\n"
        )
        await asyncio.sleep(0.05)

        # Stream namespace stats if available
        namespaces = stats.get('namespaces', {})
        if namespaces:
            yield types.TextContent(
                type="text", 
                text=f"   Namespaces: {list(namespaces.keys())}\n"
            )
            await asyncio.sleep(0.05)

        yield types.TextContent(
            type="text", 
            text="✅ Statistics retrieval completed!\n"
        )

    except Exception as e:
        yield types.TextContent(type="text", text=f"❌ Failed to retrieve stats: {str(e)}\n")


async def process_document_stream(
    arguments: dict | None, pinecone_client: PineconeClient
) -> AsyncGenerator[types.TextContent, None]:
    """Streaming document processing with progress updates"""
    try:
        document_id = arguments.get("document_id")
        text = arguments.get("text")
        namespace = arguments.get("namespace")
        metadata = arguments.get("metadata", {})
        stream = arguments.get("stream", True)

        if not stream:
            # Fallback to non-streaming version
            from .tools import process_document
            result = process_document(arguments, pinecone_client)
            for content in result:
                yield content
            return

        yield types.TextContent(
            type="text", 
            text=f"📝 Starting document processing for ID: {document_id}\n"
        )
        await asyncio.sleep(0.1)

        # Chunking step
        yield types.TextContent(
            type="text", 
            text="✂️  Chunking document...\n"
        )
        await asyncio.sleep(0.1)

        from .chunking import create_chunker
        chunker = create_chunker(chunk_type="smart")
        chunks = chunker.chunk_document(document_id, text, metadata)
        
        yield types.TextContent(
            type="text", 
            text=f"✅ Created {len(chunks)} chunks\n"
        )
        await asyncio.sleep(0.1)

        # Embedding step
        yield types.TextContent(
            type="text", 
            text="🧠 Generating embeddings...\n"
        )
        await asyncio.sleep(0.1)

        from .tools import embed_document
        embed_result = embed_document(chunks, pinecone_client)
        embedded_chunks = embed_result.get("embedded_chunks", [])
        
        yield types.TextContent(
            type="text", 
            text=f"✅ Generated embeddings for {len(embedded_chunks)} chunks\n"
        )
        await asyncio.sleep(0.1)

        # Upserting step
        yield types.TextContent(
            type="text", 
            text="📤 Uploading to Pinecone...\n"
        )
        await asyncio.sleep(0.1)

        from .tools import upsert_documents
        upsert_documents(embedded_chunks, pinecone_client, namespace)
        
        yield types.TextContent(
            type="text", 
            text="✅ Document uploaded successfully!\n"
        )
        await asyncio.sleep(0.1)

        yield types.TextContent(
            type="text", 
            text=f"🎉 Document processing completed! Document ID: {document_id}\n"
        )

    except Exception as e:
        yield types.TextContent(type="text", text=f"❌ Document processing failed: {str(e)}\n")


if __name__ == "__main__":
    asyncio.run(main()) 