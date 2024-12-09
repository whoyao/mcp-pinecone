from typing import List
from jsonschema import validate, ValidationError


def is_valid_vector_uri(uri: str) -> bool:
    """Validate vector URI format"""
    try:
        if not uri.startswith("pinecone://vectors/"):
            return False
        vector_id = uri.split("/")[-1]
        return bool(vector_id.strip())  # Ensure non-empty ID
    except Exception:
        return False


def validate_document_against_template(
    document_data: dict, template_schema: dict
) -> List[str]:
    """Validate document data against template schema with detailed errors"""
    try:
        validate(instance=document_data, schema=template_schema)
        return []
    except ValidationError as e:
        errors = []
        # ValidationError from jsonschema has different error handling
        path = " -> ".join(str(p) for p in e.path) if e.path else "root"
        errors.append(f"{path}: {e.message}")
        return errors
