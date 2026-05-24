import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict
import re
import os
from scipy.optimize import minimize

from langchain_core.tools import tool
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper


@tool
def get_best_index_for_volatility(
    target_volatility: float, test_tickers: List[str] = None
) -> Dict:
    """
    Fetches historical data from Yahoo Finance for a list of index tickers,
    calculates their 1-year realized volatility, and returns the best match.

    :param target_volatility: Target volatility as a decimal (e.g., 0.12 for 12%)
    :param test_tickers: A list of INDEX ETFs (e.g., 'SPY', 'AGG') to evaluate. Do NOT use individual company stocks.
    :return: Dict containing the best matching ticker, its actual volatility, and the error.
    """
    # Fallback to a highly diversified list across different risk profiles
    if not test_tickers:
        test_tickers = [
            "AGG",
            "SHY",
            "BNDX",  # Low Vol (~3-8%)
            "AOM",
            "AOR",  # Moderate Vol (~8-13%)
            "SPY",
            "VEA",
            "IWM",  # High Vol (~13-18%)
            "QQQ",
            "VWO",
            "INDA",  # Highest Vol (18%+)
        ]

    try:
        data = yf.download(test_tickers, period="1y")

        if data.empty:
            raise ValueError("No data returned")

        close = data["Close"]
        daily_returns = close.pct_change().dropna(how="all", axis=1).dropna()
        annualized_vol = daily_returns.std() * np.sqrt(252)

        if isinstance(annualized_vol, float):
            annualized_vol = {test_tickers[0]: annualized_vol}

        best_ticker = min(
            annualized_vol, key=lambda t: abs(annualized_vol[t] - target_volatility)
        )

        return {
            "best_matching_index": best_ticker,
            "target_volatility": target_volatility,
            "actual_volatility": round(float(annualized_vol[best_ticker]), 4),
            "difference": round(
                abs(annualized_vol[best_ticker] - target_volatility), 4
            ),
        }

    except Exception as e:
        return {"error": True, "message": str(e)}


def index_matcher_tool_mappping():
    return {"get_best_index_for_volatility": get_best_index_for_volatility}


def index_matcher_tool_list():
    tool_mapping = index_matcher_tool_mappping()
    tool_list = list(tool_mapping.values())
    return tool_list


# tools for the stock picker agent
@tool
def get_index_constituents(index_name: str) -> dict:
    """
    Returns the list of stock tickers that belong to a given index.

    Use this tool when:
    - You need all companies inside an index (e.g., S&P 500, STOXX 600)
    - You are building or filtering a portfolio from an index

    Input: index name (e.g., "S&P 500")
    Output: list of ticker symbols
    """
    search = DuckDuckGoSearchRun()

    # 1. Search specifically for the Wikipedia list or a direct data source
    search_query = f"{index_name} constituents list wikipedia"
    search_results = search.run(search_query)

    # 2. Extract URLs from the search results using regex
    # We look for Wikipedia first as it's the most structured
    urls = re.findall(r"https?://[^\s)\]]+", search_results)
    wiki_urls = [u for u in urls if "wikipedia.org" in u]

    # Use the first Wikipedia URL found, or fall back to the first result overall
    target_url = wiki_urls[0] if wiki_urls else (urls[0] if urls else None)

    if not target_url:
        return f"Could not find a reliable URL for the constituents of {index_name}."

    try:
        # 3. Use Pandas to scrape all tables from the target URL
        tables = pd.read_html(target_url)

        for df in tables:
            # Look for common column names that contain tickers
            # We add 'Symbol', 'Ticker', 'Code', and 'Identifier'
            potential_cols = [
                "Symbol",
                "Ticker",
                "Ticker symbol",
                "Component",
                "Code",
                "Company",
            ]
            found_col = next(
                (c for c in df.columns if any(p in str(c) for p in potential_cols)),
                None,
            )

            if found_col:
                # Clean the tickers: remove whitespace and handle dual-class formats (e.g., BRK.B -> BRK-B)
                tickers = (
                    df[found_col]
                    .astype(str)
                    .str.replace(r"\s+", "", regex=True)
                    .unique()
                    .tolist()
                )

                # Basic cleaning for yfinance compatibility
                clean_tickers = [t.replace(".", "-") for t in tickers if len(t) < 10]

                # Filter out header names if they were accidentally scraped
                clean_tickers = [
                    t for t in clean_tickers if t.upper() == t and t.isalpha()
                ]

                if clean_tickers:
                    return {
                        "index_name": index_name,
                        "tickers": clean_tickers,
                        "count": len(clean_tickers),
                        "source_url": target_url,
                    }

                # f"Found {len(clean_tickers)} tickers for {index_name} at {target_url}: {', '.join(clean_tickers)}..."

        return {
            "error": f"Found the page {target_url}, but couldn't find a clear ticker table. go for a duckduck go search"
        }

    except Exception as e:
        return {
            "error": f"Attempted to scrape {target_url} but failed: {str(e)}, go for a duckduckgo search"
        }


# search_tool = DuckDuckGoSearchRun()
search_tool = DuckDuckGoSearchResults(max_results=3)

wikipeida_seach_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


@tool
def get_stock_analytics(
    ticker: str, benchmark_selected: str = "SPY", riskfree_rate: float = 0.02
) -> dict:
    """
    Retrieves P/E Ratio, Beta, and calculates Alpha (relative to S&P 500) for a given ticker.
    Also returns Market Cap and Dividend Yield.
    inputs: - ticker: stock ticker symbol (e.g., AAPL)
            - benchmark_selected: the index to compare against for alpha calculation (default is SPY)
            - riskfree_rate: return from longer term government bonds to use in alpha calculation.
    output: dict with keys 'symbol', 'pe_ratio', 'beta', 'alpha_1y', 'market_cap', 'dividend_yield', 'current_price'
    """

    ticker = ticker.replace(".", "-")  # yfinance uses '-' for tickers like BRK.B

    # to write the stock price to a local file
    if os.path.exists("stock_close_prices.csv"):
        stock_close_prices = pd.read_csv(
            "stock_close_prices.csv", index_col=0, parse_dates=True
        )
    else:
        stock_close_prices = None

    # to get the stock attirbutes.
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 1. Fetch direct metrics from .info
        pe_ratio = info.get("trailingPE", "N/A")
        # beta = info.get("beta", "N/A")
        market_cap = info.get("marketCap", "N/A")
        dividend_yield = info.get("dividendYield", 0) * 100

        # 2. Calculate Alpha (Excess return over SPY)
        # We compare 1-year returns of the stock vs the benchmark
        # 1. Download both at once to ensure date alignment
        data = yf.download([ticker, benchmark_selected], period="1y")
        # check "Adj Close" is available
        if "Adj Close" in data.columns:
            data = data["Adj Close"].dropna()
        else:
            data = data["Close"].dropna()

        if len(data) > 20:
            # calculate beta
            return_data = data.pct_change().dropna()
            stock_daily_returns = return_data[ticker]
            benchmark_daily_returns = return_data[benchmark_selected]
            covarience_matrix = np.cov(stock_daily_returns, benchmark_daily_returns)
            covarience = covarience_matrix[0, 1]
            market_varience = covarience_matrix[1, 1]
            beta = covarience / market_varience

            # 2. Calculate returns using the same starting and ending dates
            returns = (data.iloc[-1] / data.iloc[0]) - 1
            stock_return = returns[ticker]
            market_return = returns[benchmark_selected]

            # 3. Standard Alpha Formula
            alpha = stock_return - (
                riskfree_rate + beta * (market_return - riskfree_rate)
            )
        else:
            return {"error": "not enough data points to calculate alpha and beta"}

        return {
            "symbol": ticker.upper(),
            "pe_ratio": pe_ratio,
            "beta": beta,
            "alpha_1y": alpha if isinstance(alpha, float) else "N/A",
            "market_cap": market_cap if isinstance(market_cap, int) else "N/A",
            "dividend_yield": dividend_yield,
            "current_price": info.get("currentPrice", "N/A"),
        }

    except Exception as e:
        return {"error": f"Failed to fetch data for {ticker}: {str(e)}"}


stock_picker_tool_mapping = {
    # "get_index_constituents": get_index_constituents,
    # search_tool.name: search_tool,
    "get_stock_analytics": get_stock_analytics,
    # wikipeida_seach_tool.name: wikipeida_seach_tool
}

stock_picker_tool_list = list(stock_picker_tool_mapping.values())


def get_stock_info(tickers):
    # 1. Fetch info for sectors
    ticker_map = {}
    for t in tickers:
        try:
            ticker_map[t] = yf.Ticker(t).info.get("sector", "Unknown")
        except:
            ticker_map[t] = "Unknown"

    # 2. Download all price data at once (Faster & Aligned)
    data = yf.download(tickers, period="1y", progress=False)

    # Handle both MultiIndex and Single Ticker cases
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        prices = data["Close"]

    prices = prices.dropna()
    daily_returns = prices.pct_change().dropna()

    return {"sector_mapping": ticker_map, "daily_returns": daily_returns}


@tool
def optimize_portfolio_weights(tickers: list[str], target_vol: float) -> dict:
    """
    Optimizes portfolio asset allocation to achieve a target annualized volatility.

    This function uses Sequential Least Squares Programming (SLSQP) to find the
    optimal weights for a given set of tickers. It minimizes the variance
    differential between the resulting portfolio, the yearly weighted return
    and a user-defined volatility target.

    Args:
        tickers (list[str]): A list of stock ticker symbols to include in the portfolio.
        target_vol (float): The desired annualized volatility (standard deviation of returns)
            expressed as a decimal (e.g., 0.15 for 15%).

    Returns:
        dict: A dictionary containing either:
            - "final_weights": A mapping of {ticker: weight} where weights sum to 1.0.
            - "error": A descriptive string if data retrieval or optimization fails.

    Constraints & Bounds:
        - Full Investment: The sum of all weights must equal 1.0 (100%).
        - Asset Bounds: Each individual ticker is constrained to a weight between 1% and 30%.
        - Sector Diversification: Total exposure to any single sector is capped at 35% to
          prevent over-concentration.
        - Risk Scaling: The optimization utilizes a 252-day scaling factor for
          annualized covariance.
    """
    try:
        n = len(tickers)
        if n == 0:
            return {"error": "No tickers provided for optimization"}

        # Basic cleaning for yfinance compatibility
        tickers = [t.replace(".", "-") for t in tickers if len(t) < 10]

        # Fetch data
        data = get_stock_info(tickers)
        returns_df = data["daily_returns"]
        sector_map = data["sector_mapping"]

        # Calculate Annualized Covariance Matrix
        # 252 represents the typical number of trading days in a year
        cov_matrix = returns_df.cov() * 252
        yearly_return = returns_df.mean() * 252

        # Objective: Minimize the square of the difference from the target volatility
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            weighted_returns = np.dot(weights, yearly_return)
            return 500 * (portfolio_vol - target_vol) ** 2 - weighted_returns

        # --- Constraints & Bounds ---

        # 1. Full Investment: All weights must sum to 100%
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

        # 2. Sector Constraints: Cap at 35% to allow solver flexibility
        # This prevents the solver from defaulting to equal weights
        sectors = set(sector_map.values())
        for sector in sectors:
            indices = [tickers.index(t) for t in tickers if sector_map.get(t) == sector]
            if indices:
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, idx=indices: 0.35 - np.sum(x[idx]),
                    }
                )

        # 3. Individual Asset Bounds: Min 1%, Max 30% per stock
        bounds = tuple((0.02, 0.20) for _ in range(n))

        # --- Optimization ---

        # Initial Guess: Start with equal weighting
        init_guess = np.array([1.0 / n] * n)

        # Execute SLSQP (Sequential Least SQuares Programming) optimization
        result = minimize(
            objective,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            return {"error": f"Optimization failed: {result.message}"}

        # Format output: {Ticker: Weight} rounded to 4 decimal places
        final_weights = {tickers[i]: round(float(result.x[i]), 4) for i in range(n)}

        return {"final_weights": final_weights}

    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


portfolio_optimizer_tool_mapping = {
    "optimize_portfolio_weights": optimize_portfolio_weights
}

portfolio_optimizer_tool_list = list(portfolio_optimizer_tool_mapping.values())
