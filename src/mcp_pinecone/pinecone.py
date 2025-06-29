from pinecone import Pinecone, ServerlessSpec, FetchResponse, UpsertResponse
from typing import List, Dict, Any, Optional, Union
import os

from pydantic import BaseModel
from .constants import (
    PINECONE_INDEX_NAME,
    PINECONE_API_KEY,
)
import logging

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer


logger = logging.getLogger(__name__)

# OpenAI embedding dimension (text-embedding-ada-002)
OPENAI_EMBEDDING_DIMENSION = 1536

# Pydantic moddel for a Pinecone record
class PineconeRecord(BaseModel):
    """
    Represents a record in Pinecone
    """

    id: str
    embedding: List[float]
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        """
        Convert to dictionary format for JSON serialization
        """
        return {
            "id": self.id,
            "embedding": self.embedding,
            "text": self.text,
            "metadata": self.metadata,
        }


class PineconeClient:
    """
    A client for interacting with Pinecone.
    """

    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
        self.embed_model = OpenAIEmbedding(api_key=openai_api_key)
        Settings.embed_model = self.embed_model

        # Initialize index after checking/creating
        self.ensure_index_exists()
        desc = self.pc.describe_index(PINECONE_INDEX_NAME)
        self.pinecone_index = self.pc.Index(
            name=PINECONE_INDEX_NAME,
            host=desc.host,  # Get the proper host from the index description
        )
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )

        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )

        # 创建响应合成器
        self.response_synthesizer = get_response_synthesizer()
        # 创建查询引擎用于生成总结
        self.query_engine = self.index.as_query_engine()

    def ensure_index_exists(self):
        """
        Check if index exists, create if it doesn't.
        """
        try:
            indexes = self.pc.list_indexes()

            exists = any(index["name"] == PINECONE_INDEX_NAME for index in indexes)
            if exists:
                logger.warning(f"Index {PINECONE_INDEX_NAME} already exists")
                return

            self.create_index()

        except Exception as e:
            logger.error(f"Error checking/creating index: {e}")
            raise

    def create_index(self):
        """
        Create a serverless index with OpenAI embedding dimension.
        """
        try:
            return self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=OPENAI_EMBEDDING_DIMENSION,
                metric="cosine",
                deletion_protection="disabled",  # Consider enabling for production
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text using OpenAI embedding model.

        Parameters:
            text: The text to generate embeddings for.

        Returns:
            List[float]: The embeddings for the text.
        """
        try:
            # Use OpenAI embedding model
            embeddings = self.embed_model.get_text_embedding(text)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings for text: {text}, error: {e}")
            raise ValueError(f"Failed to generate embeddings for text: {text}")

    def upsert_records(
        self,
        records: List[PineconeRecord],
        namespace: Optional[str] = None,
    ) -> UpsertResponse:
        """
        Upsert records into the Pinecone index.

        Parameters:
            records: List of records to upsert.
            namespace: Optional namespace to upsert into.

        Returns:
            Dict[str, Any]: The response from Pinecone.
        """
        try:
            vectors = []
            for record in records:
                # Don't continue if there's no vector values
                if not record.embedding:
                    continue

                vector_values = record.embedding
                raw_text = record.text
                record_id = record.id
                metadata = record.metadata

                logger.info(f"Record: {metadata}")

                # Add raw text to metadata
                metadata["text"] = raw_text
                vectors.append((record_id, vector_values, metadata))

            return self.index.upsert(vectors=vectors, namespace=namespace)

        except Exception as e:
            logger.error(f"Error upserting records: {e}")
            raise

    def search_records(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Search records using OpenAI embeddings.

        Parameters:
            query: The query to search for (text or pre-computed embedding vector).
            top_k: The number of results to return.
            namespace: Optional namespace to search in.
            filter: Optional filter to apply to the search.
            include_metadata: Whether to include metadata in the search results.

        Returns:
            Dict[str, Any]: The search results from Pinecone.
        """
        try:
            # If query is text, use OpenAI embedding model to get embeddings
            if top_k:
                self.retriever.similarity_top_k = top_k

            nodes = self.retriever.retrieve(query)

                        # 提取每个节点的内容和元数据
            results = []
            for node in nodes:
                content = node.get_content()
                metadata = node.metadata
                score = node.score
                source = metadata.get('google_drive_link', 'Unknown source')
                results.append({
                    'text': content,
                    'source': source,
                    'metadata': metadata,
                    'score': score,
                    'id': node.node_id,
                })
            
            return results
            
            # return self.index.query(
            #     vector=vector,
            #     top_k=top_k,
            #     namespace=namespace,
            #     include_metadata=include_metadata,
            #     filter=filter,
            # )
        except Exception as e:
            logger.error(f"Error searching records: {e}")
            raise

    def stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the index including:
        - Total vector count
        - Index dimension
        - Index fullness
        - Namespace-specific statistics

        Returns:
            Dict[str, Any]: A dictionary containing:
                - namespaces: Dict mapping namespace names to their statistics
                - dimension: Dimension of the indexed vectors
                - index_fullness: Fullness of the index (0-1 scale)
                - total_vector_count: Total number of vectors across all namespaces

        """
        try:
            stats = self.index.describe_index_stats()
            # Convert namespaces to dict - each NamespaceSummary needs to be converted to dict
            namespaces_dict = {}
            for ns_name, ns_summary in stats.namespaces.items():
                namespaces_dict[ns_name] = {
                    "vector_count": ns_summary.vector_count,
                }

            return {
                "namespaces": namespaces_dict,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "total_vector_count": stats.total_vector_count,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise

    def delete_records(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete records by ID

        Parameters:
            ids: List of record IDs to delete
            namespace: Optional namespace to delete from
        """
        try:
            return self.index.delete(ids=ids, namespace=namespace)
        except Exception as e:
            logger.error(f"Error deleting records: {e}")
            raise

    def fetch_records(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> FetchResponse:
        """
        Fetch specific records by ID

        Parameters:
            ids: List of record IDs to fetch
            namespace: Optional namespace to fetch from

        Returns:
            FetchResponse: The response from Pinecone.

        Raises:
            Exception: If there is an error fetching the records.
        """
        try:
            return self.index.fetch(ids=ids, namespace=namespace)
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            raise

    def list_records(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List records in the index using pagination.

        Parameters:
            prefix: Optional prefix to filter records by.
            limit: The number of records to return per page.
            namespace: Optional namespace to list records from.
        """
        try:
            # Using list_paginated for single-page results
            response = self.index.list_paginated(
                prefix=prefix, limit=limit, namespace=namespace
            )

            # Check if response is None
            if response is None:
                logger.error("Received None response from Pinecone list_paginated")
                return {"vectors": [], "namespace": namespace, "pagination_token": None}

            # Handle the case where vectors might be None
            vectors = response.vectors if hasattr(response, "vectors") else []

            return {
                "vectors": [
                    {
                        "id": getattr(v, "id", None),
                        "metadata": getattr(v, "metadata", {}),
                    }
                    for v in vectors
                ],
                "namespace": getattr(response, "namespace", namespace),
                "pagination_token": getattr(response.pagination, "next", None)
                if hasattr(response, "pagination")
                else None,
            }
        except Exception as e:
            logger.error(f"Error listing records: {e}")
            # Return empty result instead of raising
            return {"vectors": [], "namespace": namespace, "pagination_token": None}
