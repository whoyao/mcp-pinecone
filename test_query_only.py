#!/usr/bin/env python3
"""
Simple query test script - Only tests search/query functionality
No data insertion, just queries existing data in your Pinecone index
"""

import os
import logging
from dotenv import load_dotenv
from src.mcp_pinecone.pinecone import PineconeClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def quick_query_test():
    """Quick test of query functionality"""
    try:
        # Check environment variables
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY environment variable is required")
            return False
        
        if not os.getenv("PINECONE_API_KEY"):
            print("❌ PINECONE_API_KEY environment variable is required")
            return False
        
        print("🔍 初始化 Pinecone 客户端...")
        client = PineconeClient()
        print("✅ 客户端初始化成功")
        
        # Get index statistics first
        print("\n📊 获取索引统计信息...")
        stats = client.stats()
        total_vectors = stats.get('total_vector_count', 0)
        print(f"索引中的总向量数: {total_vectors}")
        
        if total_vectors == 0:
            print("⚠️  索引中没有数据，无法进行查询测试")
            print("请先添加一些数据到索引中，或者运行 example_usage.py 来添加测试数据")
            return False
        
        # Test queries
        test_queries = [
            "machine learning",
            "artificial intelligence", 
            "data science",
            "neural networks",
            "deep learning"
        ]
        
        print(f"\n🔍 开始查询测试 (索引中有 {total_vectors} 个向量)...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 查询 {i}: '{query}' ---")
            
            try:
                results = client.search_records(query, top_k=3)
                matches = results.get("matches", [])
                
                if matches:
                    print(f"找到 {len(matches)} 个匹配结果:")
                    for j, match in enumerate(matches, 1):
                        score = match.get("score", 0)
                        doc_id = match.get("id", "")
                        metadata = match.get("metadata", {})
                        text = metadata.get("text", "")
                        
                        # Truncate text for display
                        display_text = text[:80] + "..." if len(text) > 80 else text
                        
                        print(f"  {j}. 相似度: {score:.3f} | ID: {doc_id}")
                        print(f"     内容: {display_text}")
                        
                        # Show metadata if available
                        if metadata and len(metadata) > 1:  # More than just 'text'
                            meta_keys = [k for k in metadata.keys() if k != 'text']
                            if meta_keys:
                                print(f"     元数据: {meta_keys}")
                else:
                    print("  没有找到匹配结果")
                    
            except Exception as e:
                print(f"  ❌ 查询失败: {e}")
        
        print("\n✅ 查询测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_embedding_only():
    """Test only embedding generation without any Pinecone operations"""
    try:
        print("🧠 测试 OpenAI 嵌入生成...")
        
        client = PineconeClient()
        
        test_texts = [
            "Hello world",
            "Machine learning is fascinating",
            "这是一个中文测试"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"测试 {i}: 生成文本嵌入...")
            embeddings = client.generate_embeddings(text)
            print(f"  ✅ 生成了 {len(embeddings)} 维嵌入向量")
            print(f"  前3个值: {embeddings[:3]}")
        
        print("✅ 嵌入生成测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 嵌入生成测试失败: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("Pinecone 查询功能测试")
    print("=" * 60)
    print("此脚本只测试查询功能，不会插入新数据")
    print()
    
    # Test embedding generation first
    print("1️⃣ 测试 OpenAI 嵌入生成...")
    embedding_success = test_embedding_only()
    
    if embedding_success:
        print("\n2️⃣ 测试 Pinecone 查询功能...")
        query_success = quick_query_test()
        
        if query_success:
            print("\n🎉 所有测试通过！")
            print("\n📋 测试总结:")
            print("  ✅ OpenAI 嵌入生成: 正常")
            print("  ✅ Pinecone 查询功能: 正常")
        else:
            print("\n⚠️  查询测试失败，但嵌入生成正常")
            print("可能的原因:")
            print("  - 索引中没有数据")
            print("  - Pinecone API 配置问题")
            print("  - 网络连接问题")
    else:
        print("\n❌ 嵌入生成测试失败")
        print("请检查 OpenAI API 配置")
    
    print("\n💡 提示:")
    print("  - 如果索引为空，运行 'python example_usage.py' 添加测试数据")
    print("  - 查看详细日志: 修改 logging.basicConfig(level=logging.DEBUG)")

if __name__ == "__main__":
    main() 