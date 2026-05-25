import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, ToolMessage

# Import the orchestration nodes to test
from src.agents import (
    index_matcher,
    tool_call_node,
    tool_router,
    formatter_node,
    stock_picker,
    tool_call_node_stock_picker,
    tool_router_stock_picker,
    portfolio_optimizer,
    tool_router_portfolio_optimizer,
    formatter_node_portfolio,
)

# =====================================================================
# FIXTURES & INITIAL STATE GENERATORS
# =====================================================================


@pytest.fixture
def base_agent_state():
    """Generates a boilerplate mutable AgentState schema configuration."""
    return {
        "user_input": "Build me a low risk portfolio with $10,000",
        "chat_history": [],
        "iterations": 0,
        "base_index": "S&P 500",
        "perceived_volatility": 0.15,
        "risk_free_rate": 0.04,
        "stock_picker_history": [],
        "iterations_stock_picker": 0,
        "filtered_stocks": ["AAPL", "MSFT"],
        "portfolio_optimizer_history": [],
        "iterations_portfolio_optimizer": 0,
        "investing_sum": 10000.0,
        "risk_class": "Low",
        "expected_return": 0.08,
    }


# =====================================================================
# 1. INDEX MATCHER AGENT SUBSYSTEM TESTS
# =====================================================================


@patch("src.agents.get_llm")
@patch("src.agents.prompts")
@patch("src.agents.index_matcher_tool_list")
def test_index_matcher_under_max_calls(
    mock_tool_list, mock_prompts, mock_get_llm, base_agent_state
):
    """Verify tool binding path functions normally below boundary triggers."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Analyzing indexes...")
    mock_get_llm.return_value = mock_llm

    # FIX: Catch LangChain's | pipe operator manipulation
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_llm
    mock_prompts.index_matcher_prompt = mock_prompt
    mock_tool_list.return_value = ["mock_tool_a"]

    state = base_agent_state
    state["iterations"] = 0  # Under MAX_TOOL_CALLS boundary

    output = index_matcher(state)

    assert "chat_history" in output
    assert isinstance(output["chat_history"][0], AIMessage)
    assert output["chat_history"][0].content == "Analyzing indexes..."
    mock_llm.bind_tools.assert_called_once_with(["mock_tool_a"])


@patch("src.agents.index_matcher_tool_mappping")
def test_tool_call_node_execution(mock_tool_mapping, base_agent_state):
    """Verify that execution node loops extract parameters and invoke accurate tools."""
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = {"status": "success"}
    mock_tool_mapping.return_value = {"target_tool_name": mock_tool}

    # Simulate an upstream AI agent requesting a tool invocation sequence
    ai_message = AIMessage(content="")
    ai_message.tool_calls = [
        {
            "name": "target_tool_name",
            "args": {"target_volatility": 0.05},
            "id": "call_abc123",
        }
    ]

    state = base_agent_state
    state["chat_history"].append(ai_message)
    state["iterations"] = 1

    output = tool_call_node(state)

    assert output["iterations"] == 2
    assert len(output["chat_history"]) == 1
    assert isinstance(output["chat_history"][0], ToolMessage)
    assert output["chat_history"][0].tool_call_id == "call_abc123"
    mock_tool.invoke.assert_called_once_with({"target_volatility": 0.05})


def test_tool_router_decision_tree(base_agent_state):
    """Confirm structural conditional forks accurately dispatch graph edges."""
    state = base_agent_state

    # Path A: Unresolved open tool invocation requirements pending
    ai_msg_with_tools = AIMessage(content="")
    ai_msg_with_tools.tool_calls = [{"name": "any_tool", "args": {}, "id": "1"}]
    state["chat_history"] = [ai_msg_with_tools]
    assert tool_router(state) == "tool_call"

    # Path B: Clean return loop execution termination path
    ai_msg_clean = AIMessage(content="Final index matching report complete.")
    ai_msg_clean.tool_calls = []
    state["chat_history"] = [ai_msg_clean]
    assert tool_router(state) == "summarizer_node"


@patch("src.agents.get_llm")
@patch("src.agents.prompts")
def test_formatter_node_unwrapping(mock_prompts, mock_get_llm, base_agent_state):
    """Verify state context maps accurately into output schema data models."""
    mock_llm = MagicMock()
    mock_report = MagicMock()
    mock_report.model_dump.return_value = {
        "investing_sum": 5000.0,
        "risk_class": "Medium",
        "expected_return": 0.12,
        "base_index": "S&P 500",
        "perceived_volatility": 0.14,
        "actual_volatility": 0.135,
    }

    # Setup chain execution sequence return
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_report
    mock_llm.with_structured_output.return_value = mock_chain
    mock_get_llm.return_value = mock_llm

    # FIX: Catch LangChain's | pipe operator manipulation
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    mock_prompts.index_picker_formatter_prompt = mock_prompt

    state = base_agent_state
    state["chat_history"].append(
        AIMessage(content="Valid summary payload string context")
    )

    output = formatter_node(state)

    assert output["investing sum"] == 5000.0
    assert output["risk_class"] == "Medium"
    assert output["base_index"] == "S&P 500"


# =====================================================================
# 2. STOCK PICKER AGENT SUBSYSTEM TESTS
# =====================================================================


@patch("src.agents.get_llm")
@patch("src.agents.prompts")
def test_stock_picker_invocation(mock_prompts, mock_get_llm, base_agent_state):
    """Confirm pipeline state unpacking filters to stock analytics generator chains."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Selecting metrics...")
    mock_get_llm.return_value = mock_llm

    # FIX: Catch LangChain's | pipe operator manipulation
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_llm
    mock_prompts.stock_picker_prompt = mock_prompt

    output = stock_picker(base_agent_state)

    assert "stock_picker_history" in output
    assert output["stock_picker_history"][0].content == "Selecting metrics..."


@patch("src.agents.stock_picker_tool_mapping")
def test_tool_call_node_stock_picker_exception_handling(
    mock_tool_mapping, base_agent_state
):
    """Ensure faulty analytics tracking execution isolates tool run errors safely."""
    mock_tool = MagicMock()
    # FIX: Replaced non-existent 'RuntimeException' with 'RuntimeError'
    mock_tool.invoke.side_effect = RuntimeError(
        "YFinance API Timeout Connection Failure"
    )
    mock_tool_mapping.__contains__.return_value = True
    mock_tool_mapping.__getitem__.return_value = mock_tool

    ai_message = AIMessage(content="")
    ai_message.tool_calls = [
        {"name": "get_stock_analytics", "args": {"ticker": "AAPL"}, "id": "call_xyz"}
    ]

    state = base_agent_state
    state["stock_picker_history"].append(ai_message)
    state["iterations_stock_picker"] = 2

    output = tool_call_node_stock_picker(state)

    assert output["iterations_stock_picker"] == 3
    assert "Error calling tool" in output["stock_picker_history"][0].content
    assert output["stock_picker_history"][0].tool_call_id == "call_xyz"


def test_tool_router_stock_picker(base_agent_state):
    """Validate routing state transitions for stock picking cycles."""
    state = base_agent_state

    # Has open tool calls
    ai_msg_with_tools = AIMessage(content="")
    ai_msg_with_tools.tool_calls = [
        {"name": "get_stock_analytics", "args": {}, "id": "1"}
    ]
    state["stock_picker_history"] = [ai_msg_with_tools]
    assert tool_router_stock_picker(state) == "tool_call_node_stock_picker"

    # No tool calls left -> route to summary conversion
    ai_msg_clean = AIMessage(content="Analysis ready.")
    ai_msg_clean.tool_calls = []
    state["stock_picker_history"] = [ai_msg_clean]
    assert tool_router_stock_picker(state) == "stock_picker_summarizer"


# =====================================================================
# 3. PORTFOLIO OPTIMIZER SUBSYSTEM TESTS
# =====================================================================


@patch("src.agents.get_llm")
@patch("src.agents.prompts")
def test_portfolio_optimizer_invocation(mock_prompts, mock_get_llm, base_agent_state):
    """Confirm execution matrices feed perfectly to optimizer pipeline algorithms."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(
        content="Calculating matrix allocations..."
    )
    mock_get_llm.return_value = mock_llm

    # FIX: Catch LangChain's | pipe operator manipulation
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_llm
    mock_prompts.portfolio_optimizer_prompt = mock_prompt

    output = portfolio_optimizer(base_agent_state)

    assert "portfolio_optimizer_history" in output
    assert (
        output["portfolio_optimizer_history"][0].content
        == "Calculating matrix allocations..."
    )


def test_tool_router_portfolio_optimizer(base_agent_state):
    """Verify conditional logic forks cleanly during matrix resolution passes."""
    state = base_agent_state

    # Path A: Open processing requirements remain
    ai_msg_with_tools = AIMessage(content="")
    ai_msg_with_tools.tool_calls = [
        {"name": "optimize_portfolio_weights", "args": {}, "id": "9"}
    ]
    state["portfolio_optimizer_history"] = [ai_msg_with_tools]
    assert tool_router_portfolio_optimizer(state) == "tool_call_portfolio_optimizer"

    # Path B: Ready for summary reporting engine
    ai_msg_clean = AIMessage(content="Allocation complete.")
    ai_msg_clean.tool_calls = []
    state["portfolio_optimizer_history"] = [ai_msg_clean]
    assert tool_router_portfolio_optimizer(state) == "summarizer_portfolio_optimizer"


@patch("src.agents.get_llm")
@patch("src.agents.prompts")
def test_formatter_node_portfolio_output(mock_prompts, mock_get_llm, base_agent_state):
    """Verify that structured reporting schemas bind accurately to final allocations."""
    mock_llm = MagicMock()
    mock_report = MagicMock()

    mock_data_payload = {
        "final_weights": {"AAPL": 0.60, "MSFT": 0.40},
        "expected_portfolio_volatility": 0.11,
    }
    mock_report.model_dump.return_value = mock_data_payload

    # Setup structured output run sequence
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_report
    mock_llm.with_structured_output.return_value = mock_chain
    mock_get_llm.return_value = mock_llm

    # FIX: Catch LangChain's | pipe operator manipulation
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    mock_prompts.formatter_node_portfolio_prompt = mock_prompt

    state = base_agent_state
    state["portfolio_optimizer_history"].append(
        AIMessage(content="Allocation text report")
    )

    output = formatter_node_portfolio(state)

    assert "portfolio" in output
    assert output["portfolio"]["final_weights"]["AAPL"] == 0.60
    assert output["portfolio"]["expected_portfolio_volatility"] == 0.11
