#!/usr/bin/env python3
"""
Start script for SSE-enabled MCP Pinecone server
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from mcp_pinecone.server_sse import main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file")
        return False
    
    return True

async def start_server():
    """Start the SSE-enabled MCP server"""
    logger.info("🚀 Starting SSE-enabled MCP Pinecone server...")
    
    if not check_environment():
        logger.error("❌ Environment check failed")
        return False
    
    try:
        logger.info("✅ Environment check passed")
        logger.info("📡 Initializing server with SSE support...")
        
        # Start the server
        await main()
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return False
    
    return True

def main_sync():
    """Synchronous main function"""
    print("=" * 60)
    print("MCP Pinecone Server with SSE Support")
    print("=" * 60)
    print("Features:")
    print("  ✅ Server-Sent Events (SSE) support")
    print("  ✅ Streaming responses")
    print("  ✅ Real-time progress updates")
    print("  ✅ OpenAI embedding integration")
    print()
    
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_sync() 