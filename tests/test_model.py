import os
import pytest
from unittest.mock import patch, MagicMock

# Assuming your module is named llm_config.py inside src/
from src.model import get_llm
from src.config import api_key_var


@pytest.fixture
def mock_dependencies():
    """Fixture to mock external dependencies to avoid actual side effects."""
    with patch("src.model.config_env") as mock_config_env, patch("src.model.ChatDeepSeek") as mock_chat_deepseek:
        yield mock_config_env, mock_chat_deepseek


def test_get_llm_success(mock_dependencies):
    """Test that get_llm initializes ChatDeepSeek with the correct parameters when the API key is present."""
    mock_config_env, mock_chat_deepseek = mock_dependencies
    
    # Setup dummy environment and return value
    fake_api_key = "sk-fake-deepseek-key-12345"
    mock_instance = MagicMock()
    mock_chat_deepseek.return_value = mock_instance

    # Mock os.environ.get to return our fake key when api_key_var is requested
    with patch.dict(os.environ, {api_key_var: fake_api_key}):
        result = get_llm()

        # 1. Assert env configuration function was called
        mock_config_env.assert_called_once()

        # 2. Assert ChatDeepSeek was instantiated with the correct arguments
        mock_chat_deepseek.assert_called_once_with(
            model="deepseek-v4-flash",
            temperature=0.0,
            api_key=fake_api_key,
            extra_body={"thinking": {"type": "disabled"}}
        )

        # 3. Assert it returns the expected object instance
        assert result == mock_instance


def test_get_llm_missing_api_key(mock_dependencies):
    """Test get_llm behavior when the API key environment variable is missing."""
    mock_config_env, mock_chat_deepseek = mock_dependencies

    # Ensure the environment variable is completely absent during this test
    with patch.dict(os.environ, {}, clear=True):
        get_llm()

        # Verify it still calls the initialization, but passes None or empty to ChatDeepSeek
        mock_config_env.assert_called_once()
        mock_chat_deepseek.assert_called_once_with(
            model="deepseek-v4-flash",
            temperature=0.0,
            api_key=None,  # os.environ.get returns None if key doesn't exist
            extra_body={"thinking": {"type": "disabled"}}
        )