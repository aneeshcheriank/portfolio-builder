from typing import List, Optional
from pydantic import BaseModel, Field


# output schema for the index matcher agent
class IndexReport(BaseModel):
    investing_sum: float = Field(description="investing sum in Dollar e.g. 1000")
    risk_class: str = Field(
        description="Risk the investor is willing to take in 7 point scale ('Extremely Low', 'Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extremely High')"
    )
    expected_return: float = Field(
        description="The minimum return the user expect from this investment, e.g. 10.00"
    )
    base_index: str = Field(
        description="The ticker symbol of the selected index from the curated list, e.g., SPY, AGG."
    )
    perceived_volatility: float = Field(
        description="The perceived volatility as a decimal, e.g., 0.12."
    )
    actual_volatility: float = Field(
        description="The actual volatility of the selected index as a decimal, e.g., 0.11."
    )


# ouput schema for the stock picker agent
class Stock(BaseModel):
    ticker: str = Field(
        description="The ticker symbol of the selected stock, e.g., AAPL."
    )
    alpha: float = Field(
        description="The alpha of the stock, indicating its performance relative to the market."
    )
    beta: float = Field(
        description="The beta of the stock, indicating its volatility relative to the market."
    )
    pe_ratio: Optional[float] = Field(
        default=None,
        description="The price-to-earnings ratio of the stock, indicating its valuation."
    )


class StockSelectionReport(BaseModel):
    base_index: str = Field(
        description="The ticker symbol of the base index from which stocks were selected, e.g., SPY."
    )
    selected_stocks: List[Stock] = Field(
        description="A list of selected stocks with their respective alpha, beta, and PE ratio."
    )


# output schema for portfolio optimizer
class StockWeight(BaseModel):
    ticker: str = Field(
        description="The ticker symbol of the selected stock, e.g., AAPL."
    )
    sector: str = Field(
        description="The sector at which the stock belongs (e.g. Technology, Finance)"
    )
    ratio: float = Field(
        description="Percentage of the investable sum advised to invest in this stock e.g. 0.05"
    )


class PortfolioReport(BaseModel):
    portfolio: List[StockWeight] = Field(
        description="List of stock selected for the portfolio along with their prepotion"
    )
    # investment_sum: str = Field(description="The sum user plan to invest e.g. 1000 Euros")
    # investment_period: str = Field(description="The period at which the user plan to invest e.g. 5 years")
    # estimated_annual_return: float = Field(description="The total value of the investment at the end of the investment period e.g. 10000 Euro")
    # projected_sum_at_the_end: float = Field(description="Expected annual return e.g. .07")

class FeedbackReport(BaseModel):
    approved: bool = Field(description = "Does the user approved the preposed portfolio?")
    change_request: List[str] = Field(description = "If the user has any change request or suggestion for the preposed portfolio, it will be captured here. If not, it will be empty list.")
