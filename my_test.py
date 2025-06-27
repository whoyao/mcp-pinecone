import os
import logging
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from pinecone import Pinecone
from typing import List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

class DocumentRetriever:
    def __init__(self):
        # 初始化 embedding 模型
        self.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
        # 设置全局 embedding 模型
        Settings.embed_model = self.embed_model
        
        self.pinecone_index = self._init_pinecone()
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        
        # 使用 embed_model 创建索引
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
        
        # 创建检索器，默认返回前5个最相关的结果
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )
        # 创建响应合成器
        self.response_synthesizer = get_response_synthesizer()
        # 创建查询引擎用于生成总结
        self.query_engine = self.index.as_query_engine()

    def _init_pinecone(self):
        """初始化 Pinecone 连接"""
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = "pdf-index"
            return pc.Index(index_name)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise

    def search(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """搜索相关文档片段

        Args:
            query: 搜索查询
            top_k: 返回的最相关结果数量，如果不指定则使用默认值(5)

        Returns:
            包含相关文档片段的列表
        """
        try:
            if top_k:
                self.retriever.similarity_top_k = top_k
            
            # 获取检索结果
            nodes = self.retriever.retrieve(query)
            
            # 提取每个节点的内容和元数据
            results = []
            for node in nodes:
                content = node.get_content()
                metadata = node.metadata
                source = metadata.get('file_path', 'Unknown source')
                results.append({
                    'content': content,
                    'source': source,
                    'metadata': metadata
                })
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def summarize(self, query: Optional[str] = None) -> dict:
        """总结文档或特定查询相关的内容

        Args:
            query: 可选的查询字符串，用于过滤要总结的内容

        Returns:
            包含总结和相关文档片段的字典
        """
        try:
            if query:
                # 如果有查询，先检索相关内容
                nodes = self.retriever.retrieve(query)
                prompt = f"基于以下与'{query}'相关的内容，生成一个全面的总结。"
            else:
                # 如果没有查询，使用更大的 top_k 值获取更多内容
                original_top_k = self.retriever.similarity_top_k
                self.retriever.similarity_top_k = 10
                nodes = self.retriever.retrieve("总结所有重要内容")
                self.retriever.similarity_top_k = original_top_k
                prompt = "对所有文档内容生成一个全面的总结。"

            # 使用响应合成器生成总结
            summary = self.response_synthesizer.synthesize(prompt, nodes)
            
            # 提取相关文档片段
            source_documents = []
            for node in nodes:
                content = node.get_content()
                metadata = node.metadata
                source = metadata.get('file_path', 'Unknown source')
                source_documents.append({
                    'content': content,
                    'source': source,
                    'metadata': metadata
                })
            
            return {
                'summary': str(summary),
                'source_documents': source_documents
            }
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise

def display_search_results(results: List[dict]):
    """格式化显示搜索结果"""
    print("\n=== 搜索结果 ===")
    for i, result in enumerate(results, 1):
        print(f"\n[结果 {i}]")
        print(f"来源: {result['source']}")
        print(f"内容: {result['content'][:200]}...")  # 只显示前200个字符
        print("-" * 80)

def display_summary(summary_result: dict):
    """格式化显示总结结果"""
    print("\n=== 总结 ===")
    print(summary_result['summary'])
    print("\n=== 相关文档片段 ===")
    for i, doc in enumerate(summary_result['source_documents'], 1):
        print(f"\n[文档 {i}]")
        print(f"来源: {doc['source']}")
        print(f"内容片段: {doc['content'][:200]}...")  # 只显示前200个字符
        print("-" * 80)

def main():
    try:
        retriever = DocumentRetriever()
        
        while True:
            print("\n=== PDF文档检索系统 ===")
            print("1. 搜索特定内容")
            print("2. 总结特定主题")
            print("3. 总结所有内容")
            print("4. 退出")
            
            choice = input("\n请选择操作 (1-4): ")
            
            if choice == "1":
                query = input("请输入搜索关键词: ")
                top_k = input("请输入需要返回的结果数量 (直接回车使用默认值5): ")
                top_k = int(top_k) if top_k.strip() else None
                
                results = retriever.search(query, top_k)
                display_search_results(results)
                
            elif choice == "2":
                topic = input("请输入要总结的主题: ")
                summary_result = retriever.summarize(topic)
                display_summary(summary_result)
                
            elif choice == "3":
                summary_result = retriever.summarize()
                display_summary(summary_result)
                
            elif choice == "4":
                print("感谢使用！再见！")
                break
                
            else:
                print("无效的选择，请重试。")
                
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 