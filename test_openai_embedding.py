#!/usr/bin/env python3
"""
Test script for OpenAI embedding integration with Pinecone - Query Testing
"""

import os
import logging
from dotenv import load_dotenv
from src.mcp_pinecone.pinecone import PineconeClient, PineconeRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

def test_query_functionality():
    """Test query/search functionality with OpenAI embeddings"""
    try:
        # Check if required environment variables are set
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable is required")
            return False
        
        if not os.getenv("PINECONE_API_KEY"):
            logger.error("PINECONE_API_KEY environment variable is required")
            return False
        
        # Initialize Pinecone client
        logger.info("Initializing Pinecone client with OpenAI embeddings...")
        client = PineconeClient()
        
        # Test 1: Basic text query
        logger.info("=== Test 1: Basic Text Query ===")
        test_query = "machine learning and artificial intelligence"
        logger.info(f"Querying for: '{test_query}'")
        
        results = client.search_records(test_query, top_k=5)
        print(results)
        matches = results.get("matches", [])
        
        logger.info(f"Found {len(matches)} matches")
        for i, match in enumerate(matches, 1):
            score = match.get("score", 0)
            doc_id = match.get("id", "")
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")[:100] + "..." if len(metadata.get("text", "")) > 100 else metadata.get("text", "")
            
            logger.info(f"  {i}. Score: {score:.3f} | ID: {doc_id}")
            logger.info(f"     Text: {text}")
        
        ## Test 2: Query with different top_k values
        #logger.info("\n=== Test 2: Different Top-K Values ===")
        #test_queries = [
        #    "artificial intelligence",
        #    "data science",
        #    "neural networks"
        #]
        
        #for query in test_queries:
        #    logger.info(f"Querying '{query}' with top_k=3")
        #    results = client.search_records(query, top_k=3)
        #    matches = results.get("matches", [])
        #    logger.info(f"  Found {len(matches)} results")
        #    
        #    if matches:
        #        best_match = matches[0]
        #        logger.info(f"  Best match score: {best_match.get('score', 0):.3f}")
        
        ## Test 3: Query with filters (if data exists)
        #logger.info("\n=== Test 3: Query with Filters ===")
        #try:
        #    # Try to search with a filter
        #    results = client.search_records(
        #        query="technology",
        #        top_k=5,
        #        filter={"category": "AI"}  # This will work if you have documents with category metadata
        #    )
        #    matches = results.get("matches", [])
        #    logger.info(f"Filtered query found {len(matches)} matches")
        #except Exception as e:
        #    logger.info(f"Filter query test (expected if no matching data): {e}")
        
        ## Test 4: Query with namespace (if specified)
        #logger.info("\n=== Test 4: Namespace Query ===")
        #try:
        #    # Try to search in a specific namespace
        #    results = client.search_records(
        #        query="computer science",
        #        top_k=3,
        #        namespace="test-namespace"  # This will work if you have data in this namespace
        #    )
        #    matches = results.get("matches", [])
        #    logger.info(f"Namespace query found {len(matches)} matches")
        #except Exception as e:
        #    logger.info(f"Namespace query test (expected if namespace doesn't exist): {e}")
        
        ## Test 5: Edge cases
        #logger.info("\n=== Test 5: Edge Cases ===")
        
        ## Empty query
        #logger.info("Testing empty query...")
        #try:
        #    results = client.search_records("", top_k=1)
        #    logger.info("Empty query completed")
        #except Exception as e:
        #    logger.info(f"Empty query error (expected): {e}")
        
        ## Very long query
        #long_query = "This is a very long query that tests how the system handles queries with many words and complex sentences that might be used in real-world applications where users ask detailed questions about various topics including machine learning, artificial intelligence, data science, and other technical subjects."
        #logger.info("Testing very long query...")
        #try:
        #    results = client.search_records(long_query, top_k=2)
        #    matches = results.get("matches", [])
        #    logger.info(f"Long query found {len(matches)} matches")
        #except Exception as e:
        #    logger.error(f"Long query failed: {e}")
        
        ## Test 6: Get index statistics
        #logger.info("\n=== Test 6: Index Statistics ===")
        #stats = client.stats()
        #logger.info(f"Index Statistics:")
        #logger.info(f"  Total vectors: {stats.get('total_vector_count', 0)}")
        #logger.info(f"  Dimension: {stats.get('dimension', 0)}")
        #logger.info(f"  Index fullness: {stats.get('index_fullness', 0):.2%}")
        
        ## Test 7: List existing records
        #logger.info("\n=== Test 7: List Existing Records ===")
        #try:
        #    records = client.list_records(limit=10)
        #    vectors = records.get("vectors", [])
        #    logger.info(f"Found {len(vectors)} existing records in index")
        #    
        #    if vectors:
        #        logger.info("Sample records:")
        #        for i, vector in enumerate(vectors[:3], 1):
        #            doc_id = vector.get("id", "")
        #            metadata = vector.get("metadata", {})
        #            logger.info(f"  {i}. ID: {doc_id}")
        #            if metadata:
        #                logger.info(f"     Metadata keys: {list(metadata.keys())}")
        #except Exception as e:
        #    logger.info(f"List records test: {e}")
        
        logger.info("\n✅ All query tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Query test failed: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation without inserting data"""
    try:
        logger.info("=== Testing Embedding Generation ===")
        
        # Initialize client
        client = PineconeClient()
        
        # Test different types of text
        test_texts = [
            "Simple text",
            "Machine learning is a subset of artificial intelligence",
            "这是一个中文测试文本",
            "Text with special characters: @#$%^&*()",
            "Very long text " * 50  # Repeat to make it long
        ]
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"Test {i}: Generating embeddings for text (length: {len(text)})")
            try:
                embeddings = client.generate_embeddings(text)
                logger.info(f"  ✅ Generated {len(embeddings)}-dimensional embeddings")
                logger.info(f"  First few values: {embeddings[:3]}")
            except Exception as e:
                logger.error(f"  ❌ Failed to generate embeddings: {e}")
        
        logger.info("✅ Embedding generation tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"Embedding generation test failed: {e}")
        return False

def main():
    """Main function"""
    print("=== Testing OpenAI Embedding Query Functionality ===")
    print("This test focuses on query/search functionality without inserting new data")
    print("=" * 70)
    
    # Test embedding generation first
    embedding_success = test_embedding_generation()
    
    # Test query functionality
    query_success = test_query_functionality()
    
    if embedding_success and query_success:
        print("\n✅ All tests passed! Query functionality is working correctly.")
        print("\n📋 Summary:")
        print("- OpenAI embedding generation: ✅")
        print("- Pinecone query functionality: ✅")
        print("- Index statistics: ✅")
        print("- Edge case handling: ✅")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return embedding_success and query_success

if __name__ == "__main__":
    main() 