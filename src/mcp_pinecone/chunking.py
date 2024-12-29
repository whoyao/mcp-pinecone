from typing import List, Dict, Any, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Represents a document chunk with metadata
    """

    id: str
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        """
        Convert to dictionary format expected by embed-document
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }


class ChunkingResponse(BaseModel):
    """
    Represents the response from chunking a document
    """

    chunks: List[Chunk]
    total_chunks: int
    document_id: str
    chunk_type: str

    def to_dict(self) -> dict:
        """
        Convert response to dictionary with chunks in embed-document format
        """
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "total_chunks": self.total_chunks,
            "document_id": self.document_id,
            "chunk_type": self.chunk_type,
        }


class MarkdownChunker:
    """
    Chunks documents based on markdown structure
    Currently just on h1, h2, h3 headers
    """

    def __init__(self):
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )

    def chunk_document(
        self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split document into chunks based on markdown headers

        Parameters:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata to include with chunks

        Returns:
            List of Chunk objects with IDs and metadata
        """
        try:
            # Split based on markdown headers
            splits = self.splitter.split_text(content)
            chunks = []

            # Process each split into a chunk
            for i, split in enumerate(splits):
                # Create chunk metadata combining:
                # 1. Header hierarchy from the split
                # 2. Document metadata
                # 3. Additional passed metadata

                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_number": i + 1,
                    "total_chunks": len(splits),
                }

                # Add any additional metadata
                if metadata:
                    chunk_metadata.update(metadata)

                chunk = Chunk(
                    id=f"{document_id}#chunk{i+1}",
                    content=split.page_content,
                    metadata=chunk_metadata,
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            raise RuntimeError(f"Error chunking document: {str(e)}")
