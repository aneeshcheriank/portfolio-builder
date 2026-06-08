from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import (
    index_matcher,
    tool_call_node,
    tool_router,
    summarizer_node,
    formatter_node,
    stock_picker,
    tool_call_node_stock_picker,
    formatter_node_stock_picker,
    tool_router_stock_picker,
    stock_picker_summarizer,
    portfolio_optimizer,
    tool_call_node_portfolio_optimizer,
    tool_router_portfolio_optimizer,
    summarizer_portfolio_optimizer,
    formatter_node_portfolio,
    portfolio_explainer,
    feedback_collector,
    feedback_router
)
from src.state import AgentState


def build_graph():
    workflow = StateGraph(AgentState)

    # nodes
    workflow.add_node("index_matcher", index_matcher)
    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("summarizer_node", summarizer_node)
    workflow.add_node("formatter", formatter_node)

    # stock picker
    workflow.add_node("stock_picker", stock_picker)
    workflow.add_node("tool_call_node_stock_picker", tool_call_node_stock_picker)
    workflow.add_node("formatter_node_stock_picker", formatter_node_stock_picker)
    workflow.add_node("stock_picker_summarizer", stock_picker_summarizer)

    # portfolio optimizer
    workflow.add_node("portfolio_optimizer", portfolio_optimizer)
    workflow.add_node(
        "tool_call_portfolio_optimizer", tool_call_node_portfolio_optimizer
    )
    workflow.add_node("summarizer_portfolio_optimizer", summarizer_portfolio_optimizer)
    workflow.add_node("formatter_portfolio", formatter_node_portfolio)
    workflow.add_node("portfolio_explainer", portfolio_explainer)
    workflow.add_node("feedback_collector", feedback_collector)

    # edges
    workflow.add_edge(START, "index_matcher")
    workflow.add_edge("tool_call", "index_matcher")
    workflow.add_edge("summarizer_node", "formatter")
    workflow.add_edge("formatter", "stock_picker")
    workflow.add_edge("tool_call_node_stock_picker", "stock_picker")
    workflow.add_edge("stock_picker_summarizer", "formatter_node_stock_picker")
    workflow.add_edge("formatter_node_stock_picker", "portfolio_optimizer")
    workflow.add_edge("tool_call_portfolio_optimizer", "portfolio_optimizer")
    workflow.add_edge("summarizer_portfolio_optimizer", "formatter_portfolio")
    workflow.add_edge("formatter_portfolio", "portfolio_explainer")
    workflow.add_edge("portfolio_explainer", "feedback_collector")
    # workflow.add_edge("feedback_collector", END)

    # conditional edge
    workflow.add_conditional_edges(
        "index_matcher",
        tool_router,
        {"tool_call": "tool_call", "summarizer_node": "summarizer_node"},
    )
    # stock picker conditional edge
    workflow.add_conditional_edges(
        "stock_picker",
        tool_router_stock_picker,
        {
            "tool_call_node_stock_picker": "tool_call_node_stock_picker",
            "stock_picker_summarizer": "stock_picker_summarizer",
        },
    )

    # stock picker conditional edge
    workflow.add_conditional_edges(
        "portfolio_optimizer",
        tool_router_portfolio_optimizer,
        {
            "tool_call_portfolio_optimizer": "tool_call_portfolio_optimizer",
            "summarizer_portfolio_optimizer": "summarizer_portfolio_optimizer",
        },
    )

    # feedback conditional edge
    workflow.add_conditional_edges(
        "feedback_collector",
        feedback_router,
        {
            END: END,
            "stock_picker": "stock_picker"
        }
    )

    memory_saver = InMemorySaver()
    compiled_workflow = workflow.compile(
        checkpointer=memory_saver,
        interrupt_before=["feedback_collector"]
    )

    return compiled_workflow
