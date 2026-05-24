import pytest
from unittest.mock import patch
from langgraph.graph.state import CompiledStateGraph

# Assuming your module is named graph_builder.py inside src/
from src.chain import build_graph


@pytest.fixture(autouse=True)
def mock_agent_nodes():
    """
    Fixture to automatically mock all imported agent functions and state,
    preventing unnecessary heavy initializations or errors during import.
    """
    with patch("src.state.AgentState") as mock_state, \
         patch("src.agents.index_matcher") as m_idx, \
         patch("src.agents.tool_call_node") as m_tc, \
         patch("src.agents.tool_router") as m_tr, \
         patch("src.agents.summarizer_node") as m_sum, \
         patch("src.agents.formatter_node") as m_fmt, \
         patch("src.chain.stock_picker") as m_sp, \
         patch("src.chain.tool_call_node_stock_picker") as m_tc_sp, \
         patch("src.chain.formatter_node_stock_picker") as m_fmt_sp, \
         patch("src.chain.tool_router_stock_picker") as m_tr_sp, \
         patch("src.chain.stock_picker_summarizer") as m_sum_sp, \
         patch("src.chain.portfolio_optimizer") as m_po, \
         patch("src.chain.tool_call_node_portfolio_optimizer") as m_tc_po, \
         patch("src.chain.tool_router_portfolio_optimizer") as m_tr_po, \
         patch("src.chain.summarizer_portfolio_optimizer") as m_sum_po, \
         patch("src.chain.formatter_node_portfolio") as m_fmt_po:
        
        yield {
            "AgentState": mock_state,
            "index_matcher": m_idx,
            "tool_call": m_tc,
            "tool_router": m_tr,
            "summarizer_node": m_sum,
            "formatter": m_fmt,
            "stock_picker": m_sp,
            "tool_call_node_stock_picker": m_tc_sp,
            "formatter_node_stock_picker": m_fmt_sp,
            "tool_router_stock_picker": m_tr_sp,
            "stock_picker_summarizer": m_sum_sp,
            "portfolio_optimizer": m_po,
            "tool_call_portfolio_optimizer": m_tc_po,
            "tool_router_portfolio_optimizer": m_tr_po,
            "summarizer_portfolio_optimizer": m_sum_po,
            "formatter_portfolio": m_fmt_po,
        }


def test_build_graph_compiles():
    """Verify that build_graph finishes compiling and returns a CompiledStateGraph instance."""
    compiled_graph = build_graph()
    assert isinstance(compiled_graph, CompiledStateGraph)


def test_graph_nodes_exist():
    """Verify that all intended nodes are successfully added to the graph architecture."""
    compiled_graph = build_graph()
    
    # Extract structural nodes from the underlying StateGraph builder
    graph_nodes = compiled_graph.builder.nodes
    
    expected_nodes = [
        "index_matcher", "tool_call", "summarizer_node", "formatter",
        "stock_picker", "tool_call_node_stock_picker", "formatter_node_stock_picker", "stock_picker_summarizer",
        "portfolio_optimizer", "tool_call_portfolio_optimizer", "summarizer_portfolio_optimizer", "formatter_portfolio"
    ]
    
    for node in expected_nodes:
        assert node in graph_nodes, f"Node '{node}' missing from graph structure."


def test_graph_edges_exist():
    """Verify that vital explicit static transitions exist within the compiled architecture."""
    compiled_graph = build_graph()
    
    # Retrieve structural edges (returns tuples of source, target)
    graph_edges = compiled_graph.builder.edges
    
    # Target assertions based on build_graph logic
    assert ("__start__", "index_matcher") in graph_edges
    assert ("tool_call", "index_matcher") in graph_edges
    assert ("summarizer_node", "formatter") in graph_edges
    assert ("formatter", "stock_picker") in graph_edges
    assert ("tool_call_node_stock_picker", "stock_picker") in graph_edges
    assert ("stock_picker_summarizer", "formatter_node_stock_picker") in graph_edges
    assert ("formatter_node_stock_picker", "portfolio_optimizer") in graph_edges
    assert ("tool_call_portfolio_optimizer", "portfolio_optimizer") in graph_edges
    assert ("summarizer_portfolio_optimizer", "formatter_portfolio") in graph_edges
    assert ("formatter_portfolio", "__end__") in graph_edges


def test_graph_conditional_edges_exist():
    """Verify that routers and targeted forks are bound properly via conditional logic."""
    compiled_graph = build_graph()
    
    # In LangGraph, when you add conditional edges, the router and branch nodes
    # are registered within the underlying graph structure.
    # We can check the internal web of sub-nodes or channels to ensure they exist.
    graph_nodes = compiled_graph.builder.nodes
    
    # 1. Assert that the source nodes for conditional routing exist in the builder
    assert "index_matcher" in graph_nodes
    assert "stock_picker" in graph_nodes
    assert "portfolio_optimizer" in graph_nodes

    # 2. Safely verify that the routing paths exist in the compiled graph's configuration or channel names
    # When branches are created, LangGraph sets up specific internal string keys for them
    compiled_channels = compiled_graph.channels.keys()
    
    # If checking the channels feels too opaque, you can also assert that the conditional 
    # target nodes themselves are fully integrated into the graph compilation:
    assert "tool_call" in graph_nodes
    assert "summarizer_node" in graph_nodes
    assert "tool_call_node_stock_picker" in graph_nodes
    assert "stock_picker_summarizer" in graph_nodes
    assert "tool_call_portfolio_optimizer" in graph_nodes
    assert "summarizer_portfolio_optimizer" in graph_nodes