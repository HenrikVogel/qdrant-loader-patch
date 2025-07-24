"""Tests for configuration module."""

import os
from unittest.mock import patch

from qdrant_loader_mcp_server.config import Config, OpenAIQueryConfig, OpenAIEmbeddingConfig, QdrantConfig


def test_config_creation():
    """Test basic config creation."""
    config = Config()
    assert config is not None
    assert hasattr(config, "qdrant")
    assert hasattr(config, "openai_query")
    assert hasattr(config, "openai_embedding")


def test_qdrant_config_defaults(monkeypatch):
    """Test Qdrant configuration defaults."""
    # Clear all Qdrant-related environment variables
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_COLLECTION_NAME", raising=False)
    monkeypatch.delenv("QDRANT_VECTOR_SIZE", raising=False)

    config = QdrantConfig()
    assert config.url == "http://localhost:6333"
    assert config.collection_name == "documents"
    assert config.api_key is None
    assert config.vector_size == 4096


def test_qdrant_config_from_env(monkeypatch):
    """Test Qdrant configuration from environment variables."""
    # Set test environment variables
    monkeypatch.setenv("QDRANT_URL", "http://test:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test_key")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("QDRANT_VECTOR_SIZE", "12345")

    config = QdrantConfig()
    assert config.url == "http://test:6333"
    assert config.api_key == "test_key"
    assert config.collection_name == "test_collection"
    assert config.vector_size == 12345


def test_openai_query_config_defaults(monkeypatch):
    """Test OpenAI configuration defaults."""
    # Clear all related env variables
    monkeypatch.delenv("OPENAI_QUERY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_QUERY_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_QUERY_BASE_URL", raising=False)

    config = OpenAIQueryConfig()
    assert config.model == "gpt-3.5-turbo"
    assert config.api_key == ""
    assert config.base_url == "http://localhost:1234/v1"


def test_openai_query_config_from_env():
    """Test OpenAI configuration from environment variables."""
    with patch.dict(
        os.environ,
        {"OPENAI_QUERY_API_KEY": "test_key", "OPENAI_QUERY_MODEL": "text-embedding-ada-002", "OPENAI_QUERY_BASE_URL": "http://localhost:111234/v1"},
    ):
        config = OpenAIQueryConfig()
        assert config.api_key == "test_key"
        assert config.model == "text-embedding-ada-002"
        assert config.base_url == "http://localhost:111234/v1"

# TODO: Add unit tests for OpenAI embedding config

def test_openai__query_config_with_api_key():
    """Test OpenAI configuration with explicit API key."""
    config = OpenAIQueryConfig(api_key="explicit_key")
    assert config.api_key == "explicit_key"


def test_config_validation():
    """Test configuration validation."""
    # Test valid configuration
    qdrant_config = QdrantConfig(
        url="http://localhost:6333", collection_name="test", api_key="key"
    )
    assert qdrant_config.url == "http://localhost:6333"


def test_config_integration(monkeypatch):
    """Test full configuration integration."""
    # Set test environment variables
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "test_collection")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_QUERY_API_KEY", "test_keyaaa")

    config = Config()
    assert config.qdrant.url == "http://localhost:6333"
    assert config.qdrant.collection_name == "test_collection"
    assert config.openai_query.api_key == "test_keyaaa"
