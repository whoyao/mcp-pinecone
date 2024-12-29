import json
import logging
from typing import Union, Sequence
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from .pinecone import PineconeClient, PineconeRecord
from .utils import MCPToolError
from .chunking import MarkdownChunker, ChunkingResponse

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
            description="Search pinecone knowledge base",
            category="search",
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
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace to read from",
                    },
                },
                "required": ["document_id"],
            },
        ),
        types.Tool(
            name="chunk-document",
            description="First step in document storage process. Chunks a document into smaller segments for optimal storage and retrieval. Must be called before upsert-document.",
            category="mutation",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "text": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["document_id", "text", "metadata"],
            },
        ),
        types.Tool(
            name="embed-document",
            description="Second step in document storage process. Embeds a document into the knowledge base as a vector. Must be used after chunk-document. Expects chunks from the chunk-document response.",
            category="mutation",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                                "metadata": {"type": "object"},
                            },
                            "required": ["id", "content", "metadata"],
                        },
                    }
                },
                "required": ["chunks"],
            },
        ),
        types.Tool(
            name="upsert-document",
            description="Third step in document storage process. Upserts a document into the knowledge base. Must be used after chunk-document and embed-document. Expects embeddings from the embed-document response.",
            category="mutation",
            inputSchema={
                "type": "object",
                "properties": {
                    "embedded_chunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "embedding": {"type": "array"},
                                "text": {"type": "string"},
                                "metadata": {"type": "object"},
                            },
                            "required": ["id", "embedding", "text", "metadata"],
                        },
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace to store the document in",
                    },
                },
                "required": ["embedded_chunks"],
            },
        ),
        types.Tool(
            name="process-document",
            description="Process a document by chunking, embedding, and upserting it into the knowledge base",
            category="mutation",
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
                },
                "required": ["document_id", "text", "metadata"],
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
            namespace = arguments.get("namespace")

            results = pinecone_client.search_records(
                query=query,
                top_k=top_k,
                filter=filters,
                include_metadata=True,
                namespace=namespace,
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
            namespace = arguments.get("namespace")
            if not document_id:
                raise ValueError("document_id is required")

            # Fetch the record using your existing fetch_records method
            record = pinecone_client.fetch_records([document_id], namespace=namespace)

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

        if name == "process-document":  # New combined tool
            document_id = arguments.get("document_id")
            text = arguments.get("text")
            namespace = arguments.get("namespace")
            metadata = arguments.get("metadata", {})

            # Store a reference to the original non-chunked document id in the metadata
            metadata["original_document_id"] = document_id

            # Chain the tools together internally
            chunks_result = await handle_call_tool(
                "chunk-document",
                {"document_id": document_id, "text": text, "metadata": metadata},
            )
            chunks_data = json.loads(chunks_result[0].text)

            embed_result = await handle_call_tool(
                "embed-document", {"chunks": chunks_data["chunks"]}
            )
            embed_data = json.loads(embed_result[0].text)

            await handle_call_tool(
                "upsert-document",
                {
                    "embedded_chunks": embed_data["embedded_chunks"],
                    "namespace": namespace,
                },
            )

            return [
                types.TextContent(
                    type="text",
                    text="Successfully processed document",
                )
            ]

        elif name == "chunk-document":
            document_id = arguments.get("document_id")
            text = arguments.get("text")
            chunk_type = arguments.get("chunk_type", "markdown")
            metadata = arguments.get("metadata", {})

            chunker = MarkdownChunker()
            chunks = chunker.chunk_document(
                document_id=document_id, content=text, metadata=metadata
            )

            response = ChunkingResponse(
                chunks=chunks,
                total_chunks=len(chunks),
                document_id=document_id,
                chunk_type=chunk_type,
            )

            # Return the chunks as a list of text content
            return [types.TextContent(type="text", text=json.dumps(response.to_dict()))]

        elif name == "embed-document":
            chunks = arguments.get("chunks", [])

            embedded_chunks = []
            for chunk in chunks:
                content = chunk.get("content")
                chunk_id = chunk.get("id")
                metadata = chunk.get("metadata", {})

                if not content or not chunk_id:
                    logger.warning(f"Skipping invalid chunk: {chunk}")
                    continue

                embedding = pinecone_client.generate_embeddings(content)
                record = PineconeRecord(
                    id=chunk_id,
                    embedding=embedding,
                    text=content,
                    metadata=metadata,
                )
                embedded_chunks.append(record.to_dict())

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "embedded_chunks": embedded_chunks,
                            "total_embedded": len(embedded_chunks),
                        }
                    ),
                )
            ]

        elif name == "upsert-document":
            namespace = arguments.get("namespace")
            embedded_chunks = arguments.get("embedded_chunks", [])

            records = [PineconeRecord(**record) for record in embedded_chunks]

            result = pinecone_client.upsert_records(records, namespace=namespace)

            count = result.upserted_count or 0

            return [
                types.TextContent(
                    type="text", text=f"Successfully upserted {count} documents"
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
                server_version="0.1.5",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(resources_changed=True),
                    experimental_capabilities={},
                ),
            ),
        )
