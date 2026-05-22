from typing import TypedDict, Annotated
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
    filtered_stocks: schema.StockSelectionReport # from python 3.9 onwards, we can use list[str] instead of List[str]
    portfolio: schema.PortfolioReport

    iterations: int
    iterations_stock_picker: int
    iterations_portfolio_optimizer: int