from unittest.mock import patch
# Assuming your module is named config_env.py inside src/
from src.config_env import config_env


@patch("src.config_env.dotenv")
def test_config_env_calls_dotenv_correctly(mock_dotenv):
    """Test that config_env correctly finds and loads the .env file."""
    # Setup mock behavior: find_dotenv returns a dummy path string
    dummy_path = "/path/to/mocked/.env"
    mock_dotenv.find_dotenv.return_value = dummy_path

    # Execute the function under test
    config_env()

    # 1. Verify that find_dotenv was called to locate the .env file
    mock_dotenv.find_dotenv.assert_called_once()

    # 2. Verify that load_dotenv was called using the path returned by find_dotenv
    mock_dotenv.load_dotenv.assert_called_once_with(dummy_path)