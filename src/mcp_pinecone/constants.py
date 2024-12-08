# Index name
import os
import argparse


def get_pinecone_config():
    parser = argparse.ArgumentParser(description="Pinecone MCP Configuration")
    parser.add_argument(
        "--index-name",
        default=None,
        help="Name of the Pinecone index to use. Will use environment variable PINECONE_INDEX_NAME if not provided.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for Pinecone. Will use environment variable PINECONE_API_KEY if not provided.",
    )
    args = parser.parse_args()

    # Use command line arguments if provided, otherwise fall back to environment variables
    PINECONE_INDEX_NAME = args.index_name or os.getenv("PINECONE_INDEX_NAME")
    PINECONE_API_KEY = args.api_key or os.getenv("PINECONE_API_KEY")

    return PINECONE_INDEX_NAME, PINECONE_API_KEY


# Get configuration values
PINECONE_INDEX_NAME, PINECONE_API_KEY = get_pinecone_config()
