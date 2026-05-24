from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import operator
from langchain_core.messages import ToolMessage
from langgraph.graph import END

from src.model import get_llm
from src.tools import (
    index_matcher_tool_mappping,
    index_matcher_tool_list,
    stock_picker_tool_list,
    stock_picker_tool_mapping,
    portfolio_optimizer_tool_mapping,
    portfolio_optimizer_tool_list,
)
from src.config import MAX_TOOL_CALLS
from src.schema import IndexReport, StockSelectionReport, PortfolioReport
from src.state import AgentState
from src import prompts


def index_matcher(state: AgentState):

    prompt = prompts.index_matcher_prompt

    llm = get_llm()
    tool_list = index_matcher_tool_list()

    # need to implement a hard-stop on tool calls
    # when the limit is reached, the llm will be removed the tool calling capability and only return the final answer
    if state["iterations"] >= MAX_TOOL_CALLS:
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(tool_list)

    chain = prompt | llm_with_tools

    response = chain.invoke(state)
    return {"chat_history": [response]}


def tool_call_node(state: AgentState):
    # this node will handle the tool call and update the state accordingly
    last_state = state["chat_history"][-1]
    # increment the iteration count
    iterations = state["iterations"] + 1

    print("tool_call")

    tool_messages = []
    for tool_call in last_state.tool_calls:

        name = tool_call.get("name")
        args = tool_call.get("args")

        tool_mapping = index_matcher_tool_mappping()
        if name in tool_mapping:
            tool_response = tool_mapping[name].invoke(
                args
            )  # invoke expect dictionary as input
            tool_messages.append(
                ToolMessage(
                    content=str(tool_response), tool_call_id=tool_call.get("id")
                )
            )

    return {
        "chat_history": tool_messages,  # the tool_messages is a list
        "iterations": iterations,
    }


def tool_router(state: AgentState):
    # check the tool calls
    # if the tool call is not resolved, the agent can product its final response
    last_state = state["chat_history"][-1]
    if last_state.tool_calls:
        return "tool_call"

    # check the max_iterations
    if state["iterations"] >= MAX_TOOL_CALLS:
        return "summarizer_node"

    return "summarizer_node"


def summarizer_node(state: AgentState):
    # this node will summarize the tool calls and provide a final answer

    prompt = prompts.index_picker_summurizer_prompt

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke(
        {"context": state["chat_history"], "objective": state["user_input"]}
    )

    return {"chat_history": [response]}


def formatter_node(state: AgentState):
    context_string = state["chat_history"][-1].content if state["chat_history"] else ""

    prompt = prompts.index_picker_formatter_prompt

    llm = get_llm()
    # Structured output works best when the input is plain text context
    llm_with_structured_output = llm.with_structured_output(IndexReport)

    chain = prompt | llm_with_structured_output

    # 2. Invoke with a plain string variable instead of a message list
    response = chain.invoke({"context": context_string})
    report_data = response.model_dump()
    print(report_data)

    return {
        "chat_history": [response],
        "investig sum": report_data["investing_sum"],
        "risk_class": report_data["risk_class"],
        "expected_return": report_data["expected_return"],
        "base_index": report_data["base_index"],
        "perceived_volatility": report_data["perceived_volatility"],
        "actual_volatility": report_data["actual_volatility"],
    }


# Stock picker agent implementation
# - select the stock in the index based of alpha, beta, PE and other related factors
def stock_picker(state: AgentState):

    prompt = prompts.stock_picker_prompt

    llm = get_llm()
    llm_with_tools = llm.bind_tools(
        stock_picker_tool_list
    )  # need to define a new output schema for the stock picker
    chain = prompt | llm_with_tools
    # if state["iterations_stock_picker"] >= 5:
    #     chain = prompt | llm
    # else:
    #     chain = prompt | llm_with_tools

    response = chain.invoke(
        {
            "user_input": state["user_input"],
            "base_index": state["base_index"],
            "perceived_volatility": state["perceived_volatility"],
            "risk_free_rate": state["risk_free_rate"],
            "stock_picker_history": state["stock_picker_history"],
            "investing_sum": state.get("investing_sum"),
            "expected_return": state.get("expected_return"),
            "risk_class": state.get("risk_class"),
        }
    )

    return {"stock_picker_history": [response]}


def tool_call_node_stock_picker(state: AgentState):
    # this node will handle the tool call and update the state accordingly
    last_state = state["stock_picker_history"][-1]
    # increment the iteration count
    iterations = state["iterations_stock_picker"] + 1

    print(f"tool_call: {iterations}")

    tool_messages = []
    for tool_call in last_state.tool_calls:

        name = tool_call.get("name")
        args = tool_call.get("args")
        print(f"tool_call_name: {name}, args: {args}")

        try:
            tool_mapping = stock_picker_tool_mapping
            if name in tool_mapping:
                tool_response = tool_mapping[name].invoke(
                    args
                )  # invoke expect dictionary as input
                tool_messages.append(
                    ToolMessage(
                        content=str(tool_response), tool_call_id=tool_call.get("id")
                    )
                )
        except Exception as e:
            tool_messages.append(
                ToolMessage(
                    content=f"Error calling tool {name} with args {args}: {str(e)}",
                    tool_call_id=tool_call.get("id"),
                )
            )
    
    return {
        "stock_picker_history": tool_messages,  # the tool_messages is a list
        "iterations_stock_picker": iterations,
    }


def stock_picker_summarizer(state: AgentState) -> str:
    prompt = prompts.stock_picker_summarizer_prompt

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"chat_history": state.get("stock_picker_history")})

    return {"stock_picker_history": [response]}


def formatter_node_stock_picker(state: AgentState):
    last_message = state["stock_picker_history"][-1]

    prompt = prompts.fromatter_node_stock_picker_prompt

    llm = get_llm()
    # Structured output works best when the input is plain text context
    llm_with_structured_output = llm.with_structured_output(StockSelectionReport)

    chain = prompt | llm_with_structured_output

    # 2. Invoke with a plain string variable instead of a message list
    response = chain.invoke({"context": last_message})
    report_data = response.model_dump()

    return {
        "stock_picker_history": [response],
        "filtered_stocks": report_data["selected_stocks"],
    }


def tool_router_stock_picker(state: AgentState):
    # check the tool calls
    # if the tool call is not resolved, the agent can product its final response
    last_state = state["stock_picker_history"][-1]
    if last_state.tool_calls:
        return "tool_call_node_stock_picker"

    return "stock_picker_summarizer"


def portfolio_optimizer(state: AgentState):
    prompt = prompts.portfolio_optimizer_prompt

    llm = get_llm()
    llm_with_tools = llm.bind_tools(portfolio_optimizer_tool_list)

    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "selected_stocks": state["filtered_stocks"],
            "portfolio_optimizer_history": state["portfolio_optimizer_history"],
            "investing_sum": state["investing_sum"],
            "user_risk": state["risk_class"],
            "expected_return": state["expected_return"],
        }
    )

    return {"portfolio_optimizer_history": [response]}


def tool_call_node_portfolio_optimizer(state: AgentState):
    # this node will handle the tool call and update the state accordingly
    last_state = state["portfolio_optimizer_history"][-1]
    # increment the iteration count
    iterations = state["iterations_portfolio_optimizer"] + 1

    print(f"tool_call: {iterations}")

    tool_messages = []
    for tool_call in last_state.tool_calls:

        name = tool_call.get("name")
        args = tool_call.get("args")
        print(f"tool_call_name: {name}, args: {args}")

        try:
            tool_mapping = portfolio_optimizer_tool_mapping
            if name in tool_mapping:
                tool_response = tool_mapping[name].invoke(
                    args
                )  # invoke expect dictionary as input
                print(tool_response)
                tool_messages.append(
                    ToolMessage(
                        content=str(tool_response), tool_call_id=tool_call.get("id")
                    )
                )
        except Exception as e:
            print(f"exception in tool call: {e}")
            tool_messages.append(
                ToolMessage(
                    content=f"Error calling tool {name} with args {args}: {str(e)}",
                    tool_call_id=tool_call.get("id"),
                )
            )

    return {
        "portfolio_optimizer_history": tool_messages,  # the tool_messages is a list
        "iterations_portfolio_optimizer": iterations,
    }


def tool_router_portfolio_optimizer(state: AgentState):
    last_state = state["portfolio_optimizer_history"][-1]
    if last_state.tool_calls:
        return "tool_call_portfolio_optimizer"

    return "summarizer_portfolio_optimizer"


def summarizer_portfolio_optimizer(state: AgentState):
    # this node will summarize the tool calls and provide a final answer

    prompt = prompts.summarize_portfolio_optimizer_prompt

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"chat_history": state["portfolio_optimizer_history"]})

    return {"portfolio_optimizer_history": [response]}


def formatter_node_portfolio(state: AgentState):

    last_message = state["portfolio_optimizer_history"][-1]

    prompt = prompts.formatter_node_portfolio_prompt

    llm = get_llm()
    # Structured output works best when the input is plain text context
    llm_with_structured_output = llm.with_structured_output(PortfolioReport)

    chain = prompt | llm_with_structured_output

    # 2. Invoke with a plain string variable instead of a message list
    response = chain.invoke({"context": last_message.content})
    report_data = response.model_dump()
    return {"portfolio_optimizer_history": [response], "portfolio": report_data}
