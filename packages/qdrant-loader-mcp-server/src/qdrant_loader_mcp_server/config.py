"""Configuration settings for the RAG MCP Server."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC

# Load environment variables from .env file
load_dotenv()


class ServerConfig(BaseModel):
    """Server configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class QdrantConfig(BaseModel):
    """Qdrant configuration settings."""

    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_name: str = "documents"
    vector_size: int = 4096

    def __init__(self, **data):
        """Initialize with environment variables if not provided."""
        if "url" not in data:
            data["url"] = os.getenv("QDRANT_URL", "http://localhost:6333")
        if "api_key" not in data:
            data["api_key"] = os.getenv("QDRANT_API_KEY")
        if "collection_name" not in data:
            data["collection_name"] = os.getenv("QDRANT_COLLECTION_NAME", "documents")
        if "vector_size" not in data:
            data["vector_size"] = int(os.getenv("QDRANT_VECTOR_SIZE", "4096"))
        super().__init__(**data)


class OpenAIConfig(BaseModel, ABC):
    """Abstract base class for OpenAI configuration settings."""

    api_key: str = ""
    model: str = ""
    base_url: str = "" # if None, the OpenAI package defaults to "https://api.openai.com/v1"

class OpenAIEmbeddingConfig(OpenAIConfig):
    """OpenAI configuration for embedding."""

    def __init__(self, **data):
        """Initialize with environment variables if not provided"""
        if "api_key" not in data:
            data["api_key"] = os.getenv("OPENAI_EMBEDDING_API_KEY", "")
        if "model" not in data:
            data["model"] = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        if "base_url" not in data:
            data["base_url"] = os.getenv("OPENAI_EMBEDDING_BASE_URL", "http://localhost:1234/v1")
        super().__init__(**data)

class OpenAIQueryConfig(OpenAIConfig):
    """OpenAI configuration for queries."""

    def __init__(self, **data):
        """Initialize with environment variables if not provided"""
        if "api_key" not in data:
            data["api_key"] = os.getenv("OPENAI_QUERY_API_KEY", "")
        if "model" not in data:
            data["model"] = os.getenv("OPENAI_QUERY_MODEL", "gpt-3.5-turbo")
        if "base_url" not in data:
            data["base_url"] = os.getenv("OPENAI_QUERY_BASE_URL", "http://localhost:1234/v1")
        super().__init__(**data)


class Config(BaseModel):
    """Main configuration class."""

    server: ServerConfig
    qdrant: QdrantConfig
    openai_query: OpenAIQueryConfig
    openai_embedding: OpenAIEmbeddingConfig

    def __init__(self, **data):
        """Initialize configuration with environment variables."""
        # Initialize sub-configs if not provided
        if "server" not in data:
            data["server"] = ServerConfig()
        if "qdrant" not in data:
            data["qdrant"] = QdrantConfig()
        if "openai_query" not in data:
            data["openai_query"] = OpenAIQueryConfig()
        if "openai_embedding" not in data:
            data["openai_embedding"] = OpenAIEmbeddingConfig()
        super().__init__(**data)
