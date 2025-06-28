#!/usr/bin/env python3
"""
Test script for Server-Sent Events (SSE) functionality in MCP Pinecone server
"""

import os
import asyncio
import json
import logging
from dotenv import load_dotenv
from src.mcp_pinecone.pinecone import PineconeClient, PineconeRecord

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SSETester:
    def __init__(self):
        self.client = None
        
    def initialize(self):
        """Initialize the Pinecone client"""
        try:
            # Check environment variables
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ OPENAI_API_KEY environment variable is required")
                return False
            
            if not os.getenv("PINECONE_API_KEY"):
                print("❌ PINECONE_API_KEY environment variable is required")
                return False
            
            print("🔍 初始化 Pinecone 客户端...")
            self.client = PineconeClient()
            print("✅ 客户端初始化成功")
            
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    async def simulate_streaming_search(self, query: str, top_k: int = 5):
        """Simulate streaming search functionality"""
        print(f"\n🔍 模拟流式搜索: '{query}'")
        print("=" * 50)
        
        try:
            # Step 1: Starting search
            print("🔍 Starting semantic search...")
            await asyncio.sleep(0.5)
            
            # Step 2: Generating embeddings
            print("🧠 Generating embeddings for query...")
            await asyncio.sleep(0.5)
            
            # Step 3: Searching Pinecone
            print("📡 Searching Pinecone index...")
            await asyncio.sleep(0.5)
            
            # Perform actual search
            results = self.client.search_records(query, top_k=top_k)
            matches = results.get("matches", [])
            
            print(f"✅ Found {len(matches)} results\n")
            
            # Step 4: Stream results
            for i, match in enumerate(matches, 1):
                score = match.get("score", 0)
                doc_id = match.get("id", "")
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")[:80] + "..." if len(metadata.get("text", "")) > 80 else metadata.get("text", "")
                
                print(f"📄 Result {i}:")
                print(f"   Similarity: {score:.3f}")
                print(f"   Document ID: {doc_id}")
                print(f"   Content: {text}")
                print("-" * 40)
                
                await asyncio.sleep(0.3)  # Simulate streaming delay
            
            print("🎉 Search completed!")
            return True
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return False
    
    async def simulate_streaming_stats(self):
        """Simulate streaming stats functionality"""
        print("\n📊 模拟流式统计信息")
        print("=" * 50)
        
        try:
            print("📊 Retrieving Pinecone index statistics...")
            await asyncio.sleep(0.5)
            
            stats = self.client.stats()
            
            print("📈 Index Statistics:")
            await asyncio.sleep(0.3)
            
            # Stream each stat
            total_vectors = stats.get('total_vector_count', 0)
            print(f"   Total vectors: {total_vectors}")
            await asyncio.sleep(0.2)
            
            dimension = stats.get('dimension', 0)
            print(f"   Dimension: {dimension}")
            await asyncio.sleep(0.2)
            
            index_fullness = stats.get('index_fullness', 0)
            print(f"   Index fullness: {index_fullness:.2%}")
            await asyncio.sleep(0.2)
            
            namespaces = stats.get('namespaces', {})
            if namespaces:
                print(f"   Namespaces: {list(namespaces.keys())}")
                await asyncio.sleep(0.2)
            
            print("✅ Statistics retrieval completed!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to retrieve stats: {e}")
            return False
    
    async def simulate_streaming_document_processing(self, document_id: str, text: str):
        """Simulate streaming document processing"""
        print(f"\n📝 模拟流式文档处理: {document_id}")
        print("=" * 50)
        
        try:
            print(f"📝 Starting document processing for ID: {document_id}")
            await asyncio.sleep(0.5)
            
            # Chunking step
            print("✂️  Chunking document...")
            await asyncio.sleep(0.5)
            
            # Simulate chunking (in real implementation, this would use the chunker)
            chunks = [text[i:i+100] for i in range(0, len(text), 100)]
            print(f"✅ Created {len(chunks)} chunks")
            await asyncio.sleep(0.5)
            
            # Embedding step
            print("🧠 Generating embeddings...")
            await asyncio.sleep(0.5)
            
            # Generate embeddings for each chunk
            embeddings = []
            for i, chunk in enumerate(chunks):
                embedding = self.client.generate_embeddings(chunk)
                embeddings.append(embedding)
                if i % 2 == 0:  # Show progress every 2 chunks
                    print(f"   Processed chunk {i+1}/{len(chunks)}")
                    await asyncio.sleep(0.2)
            
            print(f"✅ Generated embeddings for {len(embeddings)} chunks")
            await asyncio.sleep(0.5)
            
            # Upserting step
            print("📤 Uploading to Pinecone...")
            await asyncio.sleep(0.5)
            
            # Create records and upsert
            records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                record = PineconeRecord(
                    id=f"{document_id}-chunk-{i}",
                    embedding=embedding,
                    text=chunk,
                    metadata={"source": document_id, "chunk_index": i}
                )
                records.append(record)
            
            self.client.upsert_records(records)
            print("✅ Document uploaded successfully!")
            await asyncio.sleep(0.5)
            
            print(f"🎉 Document processing completed! Document ID: {document_id}")
            return True
            
        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            return False
    
    async def run_sse_demo(self):
        """Run the complete SSE demo"""
        print("🚀 SSE (Server-Sent Events) 功能演示")
        print("=" * 60)
        print("此演示模拟了 MCP 服务器中的流式响应功能")
        print()
        
        if not self.initialize():
            return
        
        # Test 1: Streaming search
        await self.simulate_streaming_search("machine learning", 3)
        
        # Test 2: Streaming stats
        await self.simulate_streaming_stats()
        
        # Test 3: Streaming document processing
        test_text = """
        Machine learning is a subset of artificial intelligence that enables computers to learn 
        without being explicitly programmed. It focuses on the development of computer programs 
        that can access data and use it to learn for themselves. The process of learning begins 
        with observations or data, such as examples, direct experience, or instruction, in order 
        to look for patterns in data and make better decisions in the future based on the examples 
        that we provide.
        """
        
        await self.simulate_streaming_document_processing("demo-doc-001", test_text)
        
        print("\n🎉 SSE 演示完成！")
        print("\n💡 在实际的 MCP 服务器中，这些响应会通过 SSE 流式传输给客户端")

def main():
    """Main function"""
    tester = SSETester()
    
    # Run the async demo
    asyncio.run(tester.run_sse_demo())

if __name__ == "__main__":
    main() 