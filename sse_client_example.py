#!/usr/bin/env python3
"""
Client example for connecting to SSE-enabled MCP Pinecone server
"""

import asyncio
import json
import logging
from typing import AsyncGenerator
import aiohttp
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MCPSSEClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_streaming_tool(self, tool_name: str, arguments: dict) -> AsyncGenerator[str, None]:
        """Call a streaming tool and yield responses"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Prepare the request
        request_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        try:
            async with self.session.post(
                f"{self.server_url}/tools/call",
                json=request_data,
                headers=headers
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                # Handle SSE response
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        
                        if data == '[DONE]':
                            break
                            
                        try:
                            event_data = json.loads(data)
                            if 'result' in event_data:
                                content = event_data['result'].get('content', [])
                                for item in content:
                                    if item.get('type') == 'text':
                                        yield item.get('text', '')
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
                            
        except Exception as e:
            logger.error(f"Error calling streaming tool: {e}")
            yield f"Error: {str(e)}"
    
    async def list_tools(self) -> list:
        """List available tools"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        request_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        try:
            async with self.session.post(
                f"{self.server_url}/tools/list",
                json=request_data
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                result = await response.json()
                return result.get('result', {}).get('tools', [])
                
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []

async def demo_streaming_search():
    """Demonstrate streaming search functionality"""
    print("🔍 演示流式搜索功能")
    print("=" * 50)
    
    async with MCPSSEClient() as client:
        # List available tools
        tools = await client.list_tools()
        print(f"可用工具: {[tool['name'] for tool in tools]}")
        
        # Perform streaming search
        print("\n开始流式搜索...")
        async for chunk in client.call_streaming_tool(
            "semantic-search-stream",
            {
                "query": "machine learning",
                "top_k": 3,
                "stream": True
            }
        ):
            print(chunk, end='', flush=True)
        
        print("\n✅ 流式搜索完成")

async def demo_streaming_stats():
    """Demonstrate streaming stats functionality"""
    print("\n📊 演示流式统计功能")
    print("=" * 50)
    
    async with MCPSSEClient() as client:
        print("获取流式统计信息...")
        async for chunk in client.call_streaming_tool(
            "pinecone-stats-stream",
            {"stream": True}
        ):
            print(chunk, end='', flush=True)
        
        print("\n✅ 流式统计完成")

async def demo_streaming_document_processing():
    """Demonstrate streaming document processing"""
    print("\n📝 演示流式文档处理")
    print("=" * 50)
    
    test_document = {
        "document_id": "demo-doc-002",
        "text": """
        Deep learning is a subset of machine learning that uses neural networks with multiple layers 
        to model and understand complex patterns in data. It has been particularly successful in 
        areas such as computer vision, natural language processing, and speech recognition.
        """,
        "metadata": {
            "category": "AI",
            "source": "demo",
            "language": "en"
        },
        "stream": True
    }
    
    async with MCPSSEClient() as client:
        print("开始流式文档处理...")
        async for chunk in client.call_streaming_tool(
            "process-document-stream",
            test_document
        ):
            print(chunk, end='', flush=True)
        
        print("\n✅ 流式文档处理完成")

async def main():
    """Main function"""
    print("🚀 MCP SSE 客户端演示")
    print("=" * 60)
    print("此演示展示了如何连接到支持 SSE 的 MCP 服务器")
    print("注意：需要先启动支持 SSE 的 MCP 服务器")
    print()
    
    try:
        # Demo 1: Streaming search
        await demo_streaming_search()
        
        # Demo 2: Streaming stats
        await demo_streaming_stats()
        
        # Demo 3: Streaming document processing
        await demo_streaming_document_processing()
        
        print("\n🎉 所有 SSE 演示完成！")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        print("\n💡 请确保:")
        print("1. MCP 服务器正在运行")
        print("2. 服务器支持 SSE 功能")
        print("3. 网络连接正常")

if __name__ == "__main__":
    asyncio.run(main()) 