from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.text_splitter import MarkdownHeaderTextSplitter


@dataclass
class Chunk:
    """
    Represents a document chunk with metadata
    """

    id: str
    content: str
    metadata: Dict[str, Any]


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
        self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
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
                    "doc_id": doc_id,
                    "chunk_number": i + 1,
                    "total_chunks": len(splits),
                    "headers": split.metadata,
                    **(metadata or {}),
                }

                chunk = Chunk(
                    id=f"{doc_id}#chunk{i+1}",
                    content=split.page_content,
                    metadata=chunk_metadata,
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            raise RuntimeError(f"Error chunking document: {str(e)}")
