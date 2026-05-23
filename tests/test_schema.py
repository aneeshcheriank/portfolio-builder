import pytest
from pydantic import ValidationError
from src.schema import IndexReport, StockSelectionReport, Stock  # Adjust import based on your file name

def test_index_report_coercion():
    """Test that Pydantic correctly coerces data types (e.g., string to float)."""
    valid_data = {
        "investing_sum": "1000.50",  # Passed as string, should coerce to float
        "risk_class": "Medium",
        "expected_return": 10.0,
        "base_index": "SPY",
        "perceived_volatility": 0.12,
        "actual_volatility": 0.11
    }
    
    report = IndexReport(**valid_data)
    
    assert report.investing_sum == 1000.50  # Verifies coercion worked
    assert report.base_index == "SPY"

def test_index_report_missing_fields():
    """Test that validation fails when mandatory fields are missing."""
    incomplete_data = {
        "investing_sum": 1000.0
        # Missing all other required fields
    }
    
    with pytest.raises(ValidationError):
        IndexReport(**incomplete_data)

def test_stock_selection_nested_validation():
    """Test that nested relationships (StockSelectionReport containing Stocks) validate correctly."""
    data = {
        "base_index": "SPY",
        "selected_stocks": [
            {"ticker": "AAPL", "alpha": 1.2, "beta": 1.1, "pe_ratio": 28.5},
            {"ticker": "MSFT", "alpha": 0.9, "beta": 1.0, "pe_ratio": 32.1}
        ]
    }
    
    report = StockSelectionReport(**data)
    assert len(report.selected_stocks) == 2
    assert isinstance(report.selected_stocks[0], Stock)
    assert report.selected_stocks[0].ticker == "AAPL"