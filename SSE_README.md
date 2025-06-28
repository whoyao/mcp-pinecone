# Server-Sent Events (SSE) 支持

这个 MCP Pinecone 服务器现在支持 Server-Sent Events (SSE)，提供流式响应和实时进度更新。

## 功能特性

### 🚀 SSE 支持
- **流式响应**: 实时传输搜索结果和进度更新
- **进度指示**: 显示每个处理步骤的实时状态
- **异步处理**: 非阻塞的响应处理
- **实时反馈**: 客户端可以立即看到处理进度

### 📡 流式工具

1. **`semantic-search-stream`** - 流式语义搜索
   - 实时显示搜索进度
   - 逐个返回搜索结果
   - 显示相似度分数和元数据

2. **`pinecone-stats-stream`** - 流式统计信息
   - 逐步显示索引统计
   - 实时更新数据
   - 详细的统计信息

3. **`process-document-stream`** - 流式文档处理
   - 显示分块进度
   - 实时嵌入生成状态
   - 上传进度更新

## 快速开始

### 1. 启动 SSE 服务器

```bash
# 使用启动脚本
python start_sse_server.py

# 或直接运行
python -m src.mcp_pinecone.server_sse
```

### 2. 测试 SSE 功能

```bash
# 运行 SSE 演示
python test_sse.py

# 运行客户端示例
python sse_client_example.py
```

### 3. 查看流式响应

SSE 服务器会提供以下类型的流式响应：

```
🔍 Starting semantic search...
🧠 Generating embeddings for query...
📡 Searching Pinecone index...
✅ Found 3 results

📄 Result 1:
   Similarity: 0.892
   Document ID: doc-001
   Content: Machine learning is a subset of artificial intelligence...
----------------------------------------

📄 Result 2:
   Similarity: 0.756
   Document ID: doc-002
   Content: Deep learning uses neural networks...
----------------------------------------

🎉 Search completed!
```

## 技术实现

### 服务器端 (server_sse.py)

```python
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> AsyncGenerator[types.TextContent, None]:
    """异步生成器，支持流式响应"""
    if name == "semantic-search-stream":
        async for content in semantic_search_stream(arguments, pinecone_client):
            yield content
```

### 客户端示例

```python
async def call_streaming_tool(self, tool_name: str, arguments: dict):
    """调用流式工具并接收响应"""
    async for chunk in self.call_streaming_tool(
        "semantic-search-stream",
        {"query": "machine learning", "stream": True}
    ):
        print(chunk, end='', flush=True)
```

## 配置选项

### 环境变量

```bash
# 必需的 API 密钥
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# 可选的配置
PINECONE_INDEX_NAME=mcp-pinecone-index
DEBUG=false
```

### 工具参数

每个流式工具都支持以下参数：

- `stream`: 布尔值，控制是否启用流式响应 (默认: true)
- 其他参数与原始工具相同

## 使用示例

### 1. 流式搜索

```python
# 客户端调用
async for response in client.call_streaming_tool(
    "semantic-search-stream",
    {
        "query": "artificial intelligence",
        "top_k": 5,
        "stream": True
    }
):
    print(response, end='', flush=True)
```

### 2. 流式统计

```python
# 获取流式统计信息
async for response in client.call_streaming_tool(
    "pinecone-stats-stream",
    {"stream": True}
):
    print(response, end='', flush=True)
```

### 3. 流式文档处理

```python
# 处理文档并显示进度
async for response in client.call_streaming_tool(
    "process-document-stream",
    {
        "document_id": "my-doc",
        "text": "Document content...",
        "metadata": {"category": "AI"},
        "stream": True
    }
):
    print(response, end='', flush=True)
```

## 性能优势

### 🚀 响应速度
- **即时反馈**: 用户立即看到处理开始
- **渐进式显示**: 结果逐步显示，无需等待全部完成
- **更好的用户体验**: 避免长时间等待

### 📊 资源利用
- **异步处理**: 非阻塞的响应处理
- **内存效率**: 流式处理减少内存占用
- **可扩展性**: 支持大量并发请求

## 错误处理

### 服务器端错误处理

```python
async def semantic_search_stream(arguments, pinecone_client):
    try:
        # 处理逻辑
        yield types.TextContent(type="text", text="处理中...")
    except Exception as e:
        yield types.TextContent(type="text", text=f"❌ 错误: {str(e)}")
```

### 客户端错误处理

```python
try:
    async for response in client.call_streaming_tool(...):
        print(response, end='', flush=True)
except Exception as e:
    print(f"连接错误: {e}")
```

## 调试和监控

### 启用调试日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 监控流式响应

```python
async def monitor_streaming():
    start_time = time.time()
    chunk_count = 0
    
    async for chunk in client.call_streaming_tool(...):
        chunk_count += 1
        print(f"Chunk {chunk_count}: {len(chunk)} chars")
    
    duration = time.time() - start_time
    print(f"Total: {chunk_count} chunks in {duration:.2f}s")
```

## 最佳实践

### 1. 错误处理
- 始终包含适当的错误处理
- 提供有意义的错误消息
- 实现重试机制

### 2. 性能优化
- 使用适当的延迟时间
- 避免过大的响应块
- 监控内存使用

### 3. 用户体验
- 提供清晰的进度指示
- 使用有意义的状态消息
- 保持响应的一致性

## 故障排除

### 常见问题

1. **连接超时**
   - 检查网络连接
   - 增加超时时间
   - 验证服务器状态

2. **流式响应中断**
   - 检查客户端实现
   - 验证 SSE 支持
   - 查看服务器日志

3. **性能问题**
   - 监控资源使用
   - 优化查询参数
   - 调整延迟时间

### 调试步骤

1. 检查环境变量
2. 验证 API 密钥
3. 查看服务器日志
4. 测试网络连接
5. 验证客户端实现

## 未来改进

### 计划功能
- [ ] WebSocket 支持
- [ ] 实时通知
- [ ] 批量流式处理
- [ ] 性能监控
- [ ] 自动重连机制

### 扩展性
- [ ] 插件系统
- [ ] 自定义流式处理器
- [ ] 多租户支持
- [ ] 负载均衡

## 总结

SSE 支持为 MCP Pinecone 服务器提供了强大的流式响应能力，显著改善了用户体验和系统性能。通过实时进度更新和渐进式结果显示，用户可以更好地了解处理状态并获得更快的响应反馈。 