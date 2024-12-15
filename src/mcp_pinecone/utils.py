import time


class MCPToolError(Exception):
    """Custom exception for MCP tool errors"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def is_valid_vector_uri(uri: str) -> bool:
    """
    Validate vector URI format

    Parameters:
        uri: The URI to validate.

    Returns:
        bool: True if the URI is valid, False otherwise.s
    """
    try:
        if not uri.startswith("pinecone://vectors/"):
            return False
        vector_id = uri.split("/")[-1]
        return bool(vector_id.strip())  # Ensure non-empty ID
    except Exception:
        return False


def generate_record_id(identifier: str) -> str:
    """
    Generate a document ID using a millisecond timestamp appended to the identifier.

    Parameters:
        identifier: The identifier of the document.

    Returns:
        str: The document ID.

    Todo:
    - Implement a more robust ID generation method.
    - Store a reference to the record in a source system.

    """

    random_id = int(time.time() * 1000)
    return f"{identifier}-{random_id}"
