import pytest
from src.prompts import (
    index_matcher_prompt,
    stock_picker_prompt,
    portfolio_optimizer_prompt,
)


def test_index_matcher_prompt_variables():
    """Verify index_matcher_prompt requires the exact expected variables."""
    # .input_variables extracts variables from text and MessagesPlaceholders
    required_vars = set(index_matcher_prompt.input_variables)

    # It must expect user_input and chat_history
    assert "user_input" in required_vars
    assert "chat_history" in required_vars


def test_stock_picker_prompt_variables():
    """Ensure stock_picker_prompt has all necessary injection keys for the agent."""
    required_vars = set(stock_picker_prompt.input_variables)

    expected = {
        "base_index",
        "user_input",
        "perceived_volatility",
        "risk_class",
        "expected_return",
        "investing_sum",
        "risk_free_rate",
        "stock_picker_history",
    }

    for var in expected:
        assert (
            var in required_vars
        ), f"Missing required variable '{var}' in stock_picker_prompt"


def test_portfolio_optimizer_prompt_variables():
    """Ensure portfolio_optimizer_prompt matches the optimization node constraints."""
    required_vars = set(portfolio_optimizer_prompt.input_variables)

    assert "investing_sum" in required_vars
    assert "user_risk" in required_vars
    assert "expected_return" in required_vars
    assert "selected_stocks" in required_vars
    assert "portfolio_optimizer_history" in required_vars
