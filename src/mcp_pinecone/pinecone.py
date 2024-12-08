from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
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
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def ensure_index_exists(self):
        """Check if index exists, create if it doesn't"""
        try:
            # List all indexes
            indexes = self.pc.list_indexes()

            # Check if our index exists
            if any(index["name"] == PINECONE_INDEX_NAME for index in indexes):
                logger.info(f"Index {PINECONE_INDEX_NAME} already exists")
                return

            logger.info(f"Index {PINECONE_INDEX_NAME} not found. Creating...")
            try:
                self.create_index()
                logger.info(f"Index {PINECONE_INDEX_NAME} created successfully")
            except Exception as e:
                if "ALREADY_EXISTS" in str(e):
                    logger.info(
                        f"Index {PINECONE_INDEX_NAME} already exists (created concurrently)"
                    )
                    return
                raise

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
                deletion_protection="disabled",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    async def upsert_records(
        self, records: List[Dict[str, Any]], namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upsert records with text that will be automatically embedded

        Args:
            records: List of records with _id, text, and optional metadata
            namespace: Optional namespace to upsert into
        """
        try:
            response = await self.index.upsert_records(
                records=records, namespace=namespace
            )
            return response
        except Exception as e:
            print(f"Error upserting records: {e}")
            raise

    async def search_records(
        self,
        query: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
        fields: Optional[List[str]] = None,
        rerank: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Search records using integrated inference

        Args:
            query: Text query to search for
            top_k: Number of results to return
            namespace: Optional namespace to search in
            filter: Optional metadata filters
            fields: Optional fields to return in results
            rerank: Optional reranking configuration
        """
        try:
            response = await self.index.search_records(
                inputs=query,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                fields=fields,
                rerank=rerank,
            )
            return response
        except Exception as e:
            print(f"Error searching records: {e}")
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
            response = await self.index.delete_records(ids=ids, namespace=namespace)
            return response
        except Exception as e:
            print(f"Error deleting records: {e}")
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
            response = await self.index.fetch_records(ids=ids, namespace=namespace)
            return response
        except Exception as e:
            print(f"Error fetching records: {e}")
            raise

    async def list_records(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List records in the index

        Args:
            prefix: Optional ID prefix to filter by
            limit: Maximum number of records to return
            namespace: Optional namespace to list from
        """
        try:
            response = await self.index.list_records(
                prefix=prefix, limit=limit, namespace=namespace
            )
            return response
        except Exception as e:
            print(f"Error listing records: {e}")
            raise
