import logging
import asyncio
from typing import Union, Sequence
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from jsonschema import validate, ValidationError
from .pinecone import PineconeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pinecone-mcp")

pinecone_client = None
server = Server("pinecone-mcp")


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    try:
        records = await pinecone_client.list_records()
        return [
            types.Resource(
                uri=f"pinecone://vectors/{record['id']}",
                name=record.get("metadata", {}).get("title", f"Vector {record['id']}"),
                description=record.get("metadata", {}).get("text", "")[:100] + "...",
                metadata=record.get("metadata", {}),
                mimeType=record.get("metadata", {}).get("content_type", "text/plain"),
                template=record.get("metadata", {}).get("template_id"),
            )
            for record in records.get("vectors", [])
        ]
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        return []


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> Union[str, bytes]:
    if not str(uri).startswith("pinecone://vectors/"):
        raise ValueError(f"Unsupported URI scheme: {uri}")

    try:
        vector_id = str(uri).split("/")[-1]
        record = await pinecone_client.fetch_records([vector_id])

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


@server.list_resource_templates()
async def handle_list_resource_templates() -> list[types.ResourceTemplate]:
    return [
        types.ResourceTemplate(
            id="knowledge-note",
            name="Knowledge Note",
            uriTemplate="pinecone://vectors/{id}",
            schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "created": {"type": "string", "format": "date-time"},
                            "updated": {"type": "string", "format": "date-time"},
                            "references": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "category": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                        "required": ["title", "created"],
                    },
                },
                "required": ["text", "metadata"],
            },
        )
    ]


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="semantic-search",
            description="Search knowledge base",
            category="search",
            input_schema={
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
            name="find-similar",
            description="Find notes similar to a given note",
            category="search",
            input_schema={
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["note_id"],
            },
        ),
        types.Tool(
            name="upsert-document",
            description="Add or update a document",
            category="mutation",
            input_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "metadata": {"type": "object"},
                    "template_id": {"type": "string"},
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
            rerank = arguments.get("rerank", False)
            filters = arguments.get("filters")

            ctx = server.request_context
            await ctx.report_progress(0.2, "Searching...")

            results = await asyncio.wait_for(
                pinecone_client.search_records(
                    query=query,
                    top_k=top_k,
                    rerank={"enabled": rerank} if rerank else None,
                    filter=filters,
                ),
                timeout=30.0,
            )

            await ctx.report_progress(0.8, "Processing results...")

            matches = results.get("matches", [])
            formatted_text = f"Found {len(matches)} matches:\n" + "\n".join(
                f"- {match.get('metadata', {}).get('title', f'Document {match['id']}')} "
                f"(Score: {match['score']:.2f})"
                for match in matches
            )

            return [types.TextContent(type="text", text=formatted_text)]

        elif name == "upsert-document":
            doc_id = arguments.get("id")
            text = arguments.get("text")
            metadata = arguments.get("metadata", {})
            template_id = arguments.get("template_id")

            ctx = server.request_context
            await ctx.report_progress(0.3, "Validating document...")

            if template_id:
                templates = await handle_list_resource_templates()
                template = next((t for t in templates if t.id == template_id), None)
                if not template:
                    raise types.ToolError(f"Unknown template: {template_id}")

                document_data = {"text": text, "metadata": metadata}
                try:
                    validate(instance=document_data, schema=template.schema)
                except ValidationError as e:
                    raise types.ToolError(f"Invalid document format: {str(e)}")

            await ctx.report_progress(0.6, "Upserting document...")

            record = {
                "_id": doc_id,
                "text": text,
                "metadata": {**metadata, "template_id": template_id}
                if template_id
                else metadata,
            }

            await asyncio.wait_for(
                pinecone_client.upsert_records([record]), timeout=30.0
            )

            await ctx.report_progress(1.0, "Document upserted")

            return [
                types.TextContent(
                    type="text", text=f"Successfully upserted document: {doc_id}"
                )
            ]
        else:
            raise types.ToolError(f"Unknown tool: {name}")

    except asyncio.TimeoutError:
        raise types.ToolError("Operation timed out")
    except Exception as e:
        if isinstance(e, types.ToolError):
            raise e
        raise types.ToolError(f"Tool execution failed: {str(e)}")


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name == "brain-query":
        query = arguments.get("query")
        if not query:
            raise ValueError("Query required")

        ctx = server.request_context
        await ctx.report_progress(0.2, "Searching knowledge base...")

        # Search knowledge base
        search_results = await handle_call_tool(
            "semantic-search", {"query": query, "top_k": 5}
        )

        # Get full content of relevant notes
        notes = []
        for match in search_results[0].text.split("\n")[
            1:
        ]:  # Skip first line with count
            note_id = match.split("Document ")[1].split(" ")[0]
            content = await handle_read_resource(f"pinecone://vectors/{note_id}")
            notes.append(content)

        prompt = f"""Context from knowledge base:
{'-' * 40}
{'\n'.join(notes)}
{'-' * 40}

Question: {query}

Based on the above context:
1. Directly answer the question
2. Cite relevant information from the knowledge base
3. Note any connections between referenced documents
4. Suggest related topics to explore further"""

        return types.GetPromptResult(prompt=prompt)

    raise ValueError(f"Unknown prompt: {name}")


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
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(resources_changed=True),
                    experimental_capabilities={},
                ),
            ),
        )
