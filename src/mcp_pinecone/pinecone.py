from pinecone import Pinecone, ServerlessSpec
from typing import AsyncIterator, List, Dict, Any, Optional, Union
from .constants import PINECONE_INDEX_NAME, PINECONE_API_KEY
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class PineconeClient:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        # Initialize index after checking/creating
        self.ensure_index_exists()
        desc = self.pc.describe_index(PINECONE_INDEX_NAME)
        self.index = self.pc.Index(
            name=PINECONE_INDEX_NAME,
            host=desc.host,  # Get the proper host from the index description
        )

    def ensure_index_exists(self):
        """Check if index exists, create if it doesn't"""
        try:
            indexes = self.pc.list_indexes()

            exists = any(index["name"] == PINECONE_INDEX_NAME for index in indexes)
            if exists:
                logger.info(f"Index {PINECONE_INDEX_NAME} already exists")
                return

            logger.info(f"Index {PINECONE_INDEX_NAME} not found. Creating...")
            self.create_index()
            logger.info(f"Index {PINECONE_INDEX_NAME} created successfully")

        except Exception as e:
            logger.error(f"Error checking/creating index: {e}")
            raise

    def create_index(self):
        """Create a serverless index with integrated inference"""
        try:
            logger.info(f"Creating index {PINECONE_INDEX_NAME}")
            return self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                deletion_protection="disabled",  # Consider enabling for production
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    async def upsert_records(
        self, records: List[Dict[str, Any]], namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upsert records with text that will be automatically embedded
        """
        try:
            # Transform records to match SDK format
            vectors = [
                {
                    "id": record["_id"],
                    "values": record.get("values", []),
                    "metadata": record.get("metadata", {}),
                }
                for record in records
            ]

            return await self.index.upsert(vectors=vectors, namespace=namespace)
        except Exception as e:
            logger.error(f"Error upserting records: {e}")
            raise

    async def search_records(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Search records using integrated inference"""
        try:
            inputs = {"text": query} if isinstance(query, str) else {"values": query}
            return await self.index.query(
                **inputs,
                top_k=top_k,
                namespace=namespace,
                include_values=True,
                include_metadata=include_metadata,
                filter=filter,
            )
        except Exception as e:
            logger.error(f"Error searching records: {e}")
            raise

    async def delete_records(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete records by ID

        Args:
            ids: List of record IDs to delete
            namespace: Optional namespace to delete from
        """
        try:
            response = await self.index.delete(ids=ids, namespace=namespace)
            return response
        except Exception as e:
            logger.error(f"Error deleting records: {e}")
            raise

    async def fetch_records(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch specific records by ID

        Args:
            ids: List of record IDs to fetch
            namespace: Optional namespace to fetch from
        """
        try:
            response = await self.index.fetch(ids=ids, namespace=namespace)
            return response
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            raise

    async def list_records(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List records in the index using pagination
        """
        try:
            # Using list_paginated for single-page results
            response = await self.index.list_paginated(
                prefix=prefix, limit=limit, namespace=namespace
            )
            return {
                "vectors": [
                    {"id": v.id, "metadata": v.metadata} for v in response.vectors
                ],
                "namespace": response.namespace,
                "pagination_token": response.pagination.next
                if response.pagination
                else None,
            }
        except Exception as e:
            logger.error(f"Error listing records: {e}")  # Changed from print to logger
            raise

    # Optional: Add a method for iterating through all pages
    async def iterate_records(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> AsyncIterator[List[str]]:
        """Iterate through all records using the generator-based list method"""
        try:
            async for ids in self.index.list(
                prefix=prefix, limit=limit, namespace=namespace
            ):
                yield ids
        except Exception as e:
            logger.error(f"Error iterating records: {e}")
            raise

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        try:
            return await self.index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            raise
