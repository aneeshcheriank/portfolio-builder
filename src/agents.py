from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import operator
from langchain_core.messages import ToolMessage
from langgraph.graph import END

from src.model import get_llm
from src.tools import index_matcher_tool_mappping, index_matcher_tool_list, stock_picker_tool_list, stock_picker_tool_mapping
from src.temp import optimize_portfolio_weights
from src.config import MAX_TOOL_CALLS
from src.schema import IndexReport, StockSelectionReport, PortfolioReport
from src.temp import portfolio_optimizer_tool_mapping, portfolio_optimizer_tool_list
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
            tool_response = tool_mapping[name].invoke(args) #invoke expect dictionary as input
            tool_messages.append(
                ToolMessage(
                    content = str(tool_response),
                    tool_call_id = tool_call.get("id")
                )
            )

    return {
        "chat_history": tool_messages, #the tool_messages is a list
        "iterations": iterations
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
    response = chain.invoke({
        "chat_history": state["chat_history"],
        "user_input": state["user_input"]
    })

    return {"chat_history": [response]}    

def formatter_node(state: AgentState):
    context_string = state["chat_history"][-1].content if state["chat_history"] else ""

    prompt = prompts.index_picker_formatter_prompt
    
    llm = get_llm()
    # Structured output works best when the input is plain text context
    llm_with_structured_output = llm.with_structured_output(IndexReport)
    
    chain = prompt | llm_with_structured_output
    
    # 2. Invoke with a plain string variable instead of a message list
    response = chain.invoke({
        "context": context_string
    })
    report_data = response.model_dump()
    print(report_data)

    return {
        "chat_history": [response],
        "investig sum": report_data["investing_sum"],
        "risk_class": report_data["risk_class"],
        "expected_return": report_data["expected_return"],
        "base_index": report_data["base_index"],
        "perceived_volatility": report_data["perceived_volatility"],
        "actual_volatility": report_data["actual_volatility"]
    }

# Stock picker agent implementation
# - select the stock in the index based of alpha, beta, PE and other related factors
def stock_picker(state: AgentState):
    
    prompt = prompts.stock_picker_prompt

    llm = get_llm()
    llm_with_tools = llm.bind_tools(stock_picker_tool_list) # need to define a new output schema for the stock picker
    chain = prompt | llm_with_tools
    # if state["iterations_stock_picker"] >= 5:
    #     chain = prompt | llm
    # else:
    #     chain = prompt | llm_with_tools

    response = chain.invoke({
        "user_input": state["user_input"],
        "base_index": state["base_index"],
        "perceived_volatility": state["perceived_volatility"],
        "risk_free_rate": state["risk_free_rate"],
        "stock_picker_history": state["stock_picker_history"],
        "investing_sum": state.get("investing_sum"),
        "expected_return": state.get("expected_return"),
        "risk_class": state.get("risk_class")
    })

    return {
        "stock_picker_history": [response]
        }

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
                tool_response = tool_mapping[name].invoke(args) #invoke expect dictionary as input
                tool_messages.append(
                    ToolMessage(
                        content = str(tool_response),
                        tool_call_id = tool_call.get("id")
                    )
                )
        except Exception as e:
            tool_messages.append(
                ToolMessage(
                    content = f"Error calling tool {name} with args {args}: {str(e)}",
                    tool_call_id = tool_call.get("id")
                )
            )
    # to test the tool response
    # print(tool_messages)

    return {
        "stock_picker_history": tool_messages, #the tool_messages is a list
        "iterations_stock_picker": iterations
    }

def stock_picker_summarizer(state: AgentState)->str:
    prompt = """
    You are a financial reportor, with 10 years of experice in this field. Carefully go through the financail messages between various entites
    and summariye the converation beteen various agents and tools.
    CRITICAL:
    Please makeup any information. Only summarize the information in this conversation {chat_history}
    """

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": state.get("stock_picker_history")
    })

    return {
        "stock_picker_history": response
    }

def formatter_node_stock_picker(state: AgentState):
    # # 1. Convert the message history into a clean string for the reporter
    # # This prevents the "Unknown Tool" error and the "List vs Object" error.
    # context_string = ""
    # for msg in state["stock_picker_history"]:
    #     if hasattr(msg, 'content') and msg.content:
    #         context_string += f"{msg.type}: {msg.content}\n"
    
    # genearte ouput form the summarized output of the stock stock picker agents interactions
    last_message = state["stock_picker_history"][-1]

    prompt = prompts.fromatter_node_stock_picker_prompt
    
    llm = get_llm()
    # Structured output works best when the input is plain text context
    llm_with_structured_output = llm.with_structured_output(StockSelectionReport)
    
    chain = prompt | llm_with_structured_output
    
    # 2. Invoke with a plain string variable instead of a message list
    response = chain.invoke({
        "context": last_message
    })
    report_data = response.model_dump()

    return {
        "stock_picker_history": [response],
        "filtered_stocks": report_data["selected_stocks"]
    }

def tool_router_stock_picker(state: AgentState):    
    # check the tool calls
    # if the tool call is not resolved, the agent can product its final response
    last_state = state["stock_picker_history"][-1]
    if last_state.tool_calls:
         return "tool_call_node_stock_picker"   

    # # check the max_iterations
    # if state["iterations"] >= MAX_TOOL_CALLS:
    #     return "formatter" 
    
    return "stock_picker_summarizer"

def portfolio_optimizer(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        ### ROLE
        You are a Senior Portfolio Manager specializing in Quantitative Asset Allocation. You have 10+ years of experience in Modern Portfolio Theory and Risk 
         Parity strategies.
    
        ### OBJECTIVE
        Your goal is to allocate an investable sum {investing_sum} across a curated list of stocks to meet the client's risk and return objetives.
        The client has a risk of {user_risk} in 7 point risk scale ('Extremely Low', 'Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extremely High')
        The expected return of the client is {expected_return}
         
        ### No of stocks
        - 1000 $: 5-10 stocks
        - 10000 $: 10-20 stocks
        - 100000 $: 20-40 stocks
    
        ### INPUT DATA
        - Selected Stocks & Analytics: {selected_stocks}
    
        ### OPERATIONAL GUIDELINES
        1. **Weight Optimization**: Use the `optimize_portfolio_weights` tool. Pass the list of tickers from the selected stocks and the target volatility (as a decimal) derived from the user objective.
        2. **Concentration Risk**: Ensure no single stock exceeds 10% of the portfolio to maintain diversification.
        3. **Minimum Position Size**: Do not allocate weights less than 2% ($20) to any single stock, as small positions are inefficient.
        4. **Risk Alignment**: 
            - For Low Risk: Target Volatility ~0.05 - 0.08
            - For Moderate Risk: Target Volatility ~0.09 - 0.13
            - For High Risk: Target Volatility ~0.14 - 0.20
    
        ### EXECUTION FLOW
        - First, call `optimize_portfolio_weights` with the appropriate parameters.
        - Once you receive the tool output, verify that the weights sum to 100%.
        - If the tool fails, suggest an equal-weighted portfolio as a fallback and explain why.
        """),
        MessagesPlaceholder(variable_name="portfolio_optimizer_history")
    ])

    llm = get_llm()
    llm_with_tools = llm.bind_tools(portfolio_optimizer_tool_list)

    chain = prompt | llm_with_tools
    response = chain.invoke({
        "selected_stocks": state["filtered_stocks"],
        "portfolio_optimizer_history": state["portfolio_optimizer_history"],
        "investing_sum": state["investing_sum"],
        "user_risk": state["risk_class"],
        "expected_return": state["expected_return"] 
    })

    return{
        "portfolio_optimizer_history": [response]
    }

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
                tool_response = tool_mapping[name].invoke(args) #invoke expect dictionary as input
                print(tool_response)
                tool_messages.append(
                    ToolMessage(
                        content = str(tool_response),
                        tool_call_id = tool_call.get("id")
                    )
                )
        except Exception as e:
            print(f"exception in tool call: {e}")
            tool_messages.append(
                ToolMessage(
                    content = f"Error calling tool {name} with args {args}: {str(e)}",
                    tool_call_id = tool_call.get("id")
                )
            )
    # to test the tool response
    # print(tool_messages)

    return {
        "portfolio_optimizer_history": tool_messages, #the tool_messages is a list
        "iterations_portfolio_optimizer": iterations
    }

def tool_router_portfolio_optimizer(state: AgentState):
    last_state = state["portfolio_optimizer_history"][-1]
    if last_state.tool_calls:
         return "tool_call_portfolio_optimizer"
    
    return "summarizer_portfolio_optimizer"

def summarizer_portfolio_optimizer(state: AgentState):
    # this node will summarize the tool calls and provide a final answer

    prompt = ChatPromptTemplate.from_messages([
        ("system",
     """You are an expert financial reporter. You have multiple years of experience in financial analysis and reporting. Your task is to take 
     the output from the previous tool calls, which messages to find the best portfolio weights for a target volatility.
     - Summarize the details (do not include extra information) for futher processing.
     - capture all informations
     CRITICAL: 
     - You must use the EXACT weights provided in the last ToolMessage. 
     - DO NOT recalculate, round differently, or equalize the weights. 
     - If the tool says AAPL is 0.0829, you must report 0.0829.
     """),
        MessagesPlaceholder(variable_name="chat_history")
    ])

    
    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": state["portfolio_optimizer_history"]
    })

    return {"portfolio_optimizer_history": [response]}   

def formatter_node_portfolio(state: AgentState):
   
   last_message = state["portfolio_optimizer_history"][-1]

   prompt = ChatPromptTemplate.from_messages([
       ("system",
        """You are an expert financial reporter. 
        Take the following context (User goals and Tool results) and 
        generate the final IndexReport.
        """),
       ("human", "Here is the investment context:\n\n{context}")
   ])
   
   llm = get_llm()
   # Structured output works best when the input is plain text context
   llm_with_structured_output = llm.with_structured_output(PortfolioReport)
   
   chain = prompt | llm_with_structured_output
   
   # 2. Invoke with a plain string variable instead of a message list
   response = chain.invoke({
       "context": last_message.content
   })
   report_data = response.model_dump()
   return {
       "portfolio_optimizer_history": [response],
       "portfolio": report_data
   }
