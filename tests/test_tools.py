import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Assuming your file is named tools.py inside src/
from src.tools import (
    get_best_index_for_volatility,
    get_index_constituents,
    get_stock_analytics,
    optimize_portfolio_weights
)


# =====================================================================
# 1. TESTS FOR: get_best_index_for_volatility
# =====================================================================

@patch("src.tools.yf.download")
def test_get_best_index_for_volatility_success(mock_download):
    """Verify that it calculates annualized volatility and picks the closest index."""
    # Create mock price data for two dummy assets
    # Asset A has zero variance, Asset B has huge variance
    dates = pd.date_range(start="2025-01-01", periods=5)
    mock_df = pd.DataFrame({
        ("Close", "LOW_VOL"): [10, 10.01, 10, 10.01, 10],
        ("Close", "HIGH_VOL"): [10, 15, 8, 20, 5]
    }, index=dates)
    mock_download.return_value = mock_df

    # Target low volatility
    result = get_best_index_for_volatility.invoke({"target_volatility": 0.02, "test_tickers":["LOW_VOL", "HIGH_VOL"]})
    
    assert "error" not in result
    assert result["best_matching_index"] == "LOW_VOL"
    assert isinstance(result["actual_volatility"], float)


@patch("src.tools.yf.download")
def test_get_best_index_for_volatility_empty_data(mock_download):
    """Ensure it handles empty yfinance DataFrames gracefully."""
    mock_download.return_value = pd.DataFrame()  # Empty dataframe
    
    result = get_best_index_for_volatility.invoke({"target_volatility": 0.12})
    assert result["error"] is True
    assert "No data returned" in result["message"]


# =====================================================================
# 2. TESTS FOR: get_index_con
# =====================================================================

@patch("src.tools.pd.read_html")
@patch("src.tools.DuckDuckGoSearchRun")
def test_get_index_constituents_success(mock_ddg_class, mock_read_html):
    """Test successful scraping of Wikipedia tables for ticker arrays."""
    # Mock DuckDuckGo behavior
    mock_search_instance = MagicMock()
    mock_search_instance.run.return_value = "Here is a link https://en.wikipedia.org/wiki/S%26P_500_components"
    mock_ddg_class.return_value = mock_search_instance

    # Mock Pandas table scraping return matrix
    mock_table = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "GOOGL"]
    })
    mock_read_html.return_value = [mock_table]

    result = get_index_constituents.invoke({"index_name": "S&P 500"})

    assert result["index_name"] == "S&P 500"
    assert "AAPL" in result["tickers"]
    assert result["count"] == 3


@patch("src.tools.DuckDuckGoSearchRun")
def test_get_index_constituents_no_urls(mock_ddg_class):
    """Ensure a helpful fallback string is returned if search hits a dead end."""
    mock_search_instance = MagicMock()
    mock_search_instance.run.return_value = "No results found or links anywhere."
    mock_ddg_class.return_value = mock_search_instance

    result = get_index_constituents.invoke({"index_name": "Obscure Index Name"})
    assert "Could not find a reliable URL" in result


# =====================================================================
# 3. TESTS FOR: get_stock_analytics
# =====================================================================

@patch("src.tools.yf.download")
@patch("src.tools.yf.Ticker")
def test_get_stock_analytics_success(mock_ticker_class, mock_download):
    """Verify statistical metric retrieval and capital assets formula engine."""
    # 1. Mock the Ticker .info dict metadata attributes
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {
        "trailingPE": 25.5,
        "marketCap": 2000000000,
        "dividendYield": 0.015,
        "currentPrice": 150.0
    }
    mock_ticker_class.return_value = mock_ticker_instance

    # 2. Mock 1-year daily adjusted closing price arrays (Stock vs Benchmark)
    dates = pd.date_range(start="2025-01-01", periods=30)
    # Generate static trends to guarantee variance math processes cleanly
    mock_prices = pd.DataFrame({
        "AAPL": np.linspace(100, 120, 30),
        "SPY": np.linspace(400, 440, 30)
    }, index=dates)
    mock_prices.columns.name = None
    mock_download.return_value = mock_prices

    result = get_stock_analytics.invoke({"symbol": "AAPL", "benchmark_selected": "SPY", "riskfree_rate": 0.02})

    assert result["symbol"] == "AAPL"
    assert result["pe_ratio"] == 25.5
    assert "beta" in result
    assert "alpha_1y" in result
    assert result["dividend_yield"] == 1.5  # 0.015 * 100


# =====================================================================
# 4. TESTS FOR: optimize_portfolio_weights
# =====================================================================

@patch("src.tools.get_stock_info")
def test_optimize_portfolio_weights_success(mock_get_stock_info):
    """Test standard optimization processing and configuration boundary integrity."""
    # Set up returns across 10 steps to compute static matrices
    dates = pd.date_range(start="2025-01-01", periods=10)
    mock_returns_df = pd.DataFrame({
        "TICKER_A": [0.01, -0.005, 0.02, 0.01, -0.01, 0.005, 0.012, -0.002, 0.01, 0.015],
        "TICKER_B": [-0.01, 0.02, -0.015, 0.01, 0.03, -0.01, 0.005, 0.022, -0.01, 0.025]
    }, index=dates)

    mock_get_stock_info.return_value = {
        "sector_mapping": {"TICKER_A": "Tech", "TICKER_B": "Finance"},
        "daily_returns": mock_returns_df
    }

    result = optimize_portfolio_weights(["TICKER_A", "TICKER_B"], target_vol=0.12)

    assert "final_weights" in result
    weights = result["final_weights"]
    
    # Assert total investment allocation sums exactly close to 100% capacity
    assert pytest.approx(sum(weights.values()), rel=1e-3) == 1.0
    
    # Check asset boundary constraints (each asset bounded between 2% and 20%)
    for ticker, weight in weights.items():
        assert 0.02 <= weight <= 0.20


def test_optimize_portfolio_weights_empty_input():
    """Fail directly without calling calculations if target array is blank."""
    result = optimize_portfolio_weights([], target_vol=0.15)
    assert "error" in result
    assert "No tickers provided" in result["error"]