#!/usr/bin/env python3
"""
Example usage of OpenAI embedding integration with Pinecone
"""

import os
import logging
from dotenv import load_dotenv
from src.mcp_pinecone.pinecone import PineconeClient, PineconeRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def example_basic_usage():
    """Example of basic usage with OpenAI embeddings"""
    print("=== Basic Usage Example ===")
    
    # Initialize client
    client = PineconeClient()
    
    # Sample documents
    documents = [
        {
            "id": "doc-001",
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "metadata": {"category": "AI", "source": "tutorial"}
        },
        {
            "id": "doc-002", 
            "text": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "metadata": {"category": "AI", "source": "research"}
        },
        {
            "id": "doc-003",
            "text": "Natural language processing helps computers understand, interpret, and generate human language.",
            "metadata": {"category": "NLP", "source": "guide"}
        }
    ]
    
    # Process each document
    records = []
    for doc in documents:
        # Generate embeddings using OpenAI
        embeddings = client.generate_embeddings(doc["text"])
        
        # Create Pinecone record
        record = PineconeRecord(
            id=doc["id"],
            embedding=embeddings,
            text=doc["text"],
            metadata=doc["metadata"]
        )
        records.append(record)
    
    # Upsert to Pinecone
    print("Upserting documents...")
    client.upsert_records(records)
    print("Documents upserted successfully!")
    
    return client

def example_search(client):
    """Example of searching with OpenAI embeddings"""
    print("\n=== Search Example ===")
    
    # Search queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Tell me about natural language processing"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = client.search_records(query, top_k=3)
        
        matches = results.get("matches", [])
        for i, match in enumerate(matches, 1):
            score = match.get("score", 0)
            doc_id = match.get("id", "")
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")[:100] + "..." if len(metadata.get("text", "")) > 100 else metadata.get("text", "")
            
            print(f"  {i}. Score: {score:.3f} | ID: {doc_id}")
            print(f"     Text: {text}")
            print(f"     Category: {metadata.get('category', 'N/A')}")

def example_advanced_search(client):
    """Example of advanced search with filters"""
    print("\n=== Advanced Search Example ===")
    
    # Search with filter
    print("Searching for AI-related documents...")
    results = client.search_records(
        query="artificial intelligence",
        top_k=5,
        filter={"category": "AI"}
    )
    
    matches = results.get("matches", [])
    print(f"Found {len(matches)} AI-related documents:")
    
    for i, match in enumerate(matches, 1):
        score = match.get("score", 0)
        doc_id = match.get("id", "")
        metadata = match.get("metadata", {})
        print(f"  {i}. Score: {score:.3f} | ID: {doc_id} | Source: {metadata.get('source', 'N/A')}")

def example_index_management(client):
    """Example of index management operations"""
    print("\n=== Index Management Example ===")
    
    # Get index statistics
    stats = client.stats()
    print(f"Index Statistics:")
    print(f"  Total vectors: {stats.get('total_vector_count', 0)}")
    print(f"  Dimension: {stats.get('dimension', 0)}")
    print(f"  Index fullness: {stats.get('index_fullness', 0):.2%}")
    
    # List records
    print(f"\nListing records:")
    records = client.list_records(limit=10)
    vectors = records.get("vectors", [])
    print(f"  Found {len(vectors)} records:")
    
    for i, vector in enumerate(vectors, 1):
        doc_id = vector.get("id", "")
        metadata = vector.get("metadata", {})
        category = metadata.get("category", "N/A")
        print(f"    {i}. ID: {doc_id} | Category: {category}")

def main():
    """Main function demonstrating OpenAI embedding integration"""
    print("OpenAI Embedding Integration with Pinecone - Example Usage")
    print("=" * 60)
    
    try:
        # Check environment variables
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY environment variable is required")
            return
        
        if not os.getenv("PINECONE_API_KEY"):
            print("❌ PINECONE_API_KEY environment variable is required")
            return
        
        # Run examples
        client = example_basic_usage()
        example_search(client)
        example_advanced_search(client)
        example_index_management(client)
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    main() 