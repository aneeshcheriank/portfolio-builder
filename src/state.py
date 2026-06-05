from typing import TypedDict, Annotated # Sequence[BaseMessage] to replace list
from langgraph.graph.message import add_messages #replaced opeator.add
import operator

from src import schema


class AgentState(TypedDict):
    user_input: str

    chat_history: Annotated[list, operator.add]
    stock_picker_history: Annotated[list, operator.add]
    portfolio_optimizer_history: Annotated[list, operator.add]

    investing_sum: float
    risk_class: str
    expected_return: float
    perceived_volatility: float
    actual_volatility: float
    base_index: str
    risk_free_rate: float
    filtered_stocks: schema.StockSelectionReport
    portfolio: schema.PortfolioReport

    # Human in the loop feedback
    explanation: str
    feedback: str
    approval: bool

    iterations: int
    iterations_stock_picker: int
    iterations_portfolio_optimizer: int
