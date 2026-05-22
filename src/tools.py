import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict
import re
import os

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

@tool
def get_best_index_for_volatility(target_volatility: float, test_tickers: List[str]=None) -> Dict:
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
            "AGG", "SHY", "BNDX",  # Low Vol (~3-8%)
            "AOM", "AOR",          # Moderate Vol (~8-13%)
            "SPY", "VEA", "IWM",   # High Vol (~13-18%)
            "QQQ", "VWO", "INDA"   # Highest Vol (18%+)
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
            annualized_vol,
            key=lambda t: abs(annualized_vol[t] - target_volatility)
        )

        return {
            "best_matching_index": best_ticker,
            "target_volatility": target_volatility,
            "actual_volatility": round(float(annualized_vol[best_ticker]), 4),
            "difference": round(abs(annualized_vol[best_ticker] - target_volatility), 4)
        }

    except Exception as e:
        return {
            "error": True,
            "message": str(e)
        }


def index_matcher_tool_mappping():
    return {
        "get_best_index_for_volatility": get_best_index_for_volatility
    }

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
    urls = re.findall(r'https?://[^\s)\]]+', search_results)
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
            potential_cols = ['Symbol', 'Ticker', 'Ticker symbol', 'Component', 'Code', 'Company']
            found_col = next((c for c in df.columns if any(p in str(c) for p in potential_cols)), None)
            
            if found_col:
                # Clean the tickers: remove whitespace and handle dual-class formats (e.g., BRK.B -> BRK-B)
                tickers = df[found_col].astype(str).str.replace(r'\s+', '', regex=True).unique().tolist()
                
                # Basic cleaning for yfinance compatibility
                clean_tickers = [t.replace('.', '-') for t in tickers if len(t) < 10] 
                
                # Filter out header names if they were accidentally scraped
                clean_tickers = [t for t in clean_tickers if t.upper() == t and t.isalpha()]
                
                if clean_tickers:
                    return {
                        "index_name": index_name,
                        "tickers": clean_tickers,
                        "count": len(clean_tickers),
                        "source_url": target_url
                    }
                
                # f"Found {len(clean_tickers)} tickers for {index_name} at {target_url}: {', '.join(clean_tickers)}..."
        
        return {"error": f"Found the page {target_url}, but couldn't find a clear ticker table. go for a duckduck go search"}
        
    except Exception as e:
        return {"error": f"Attempted to scrape {target_url} but failed: {str(e)}, go for a duckduckgo search"}
    
# search_tool = DuckDuckGoSearchRun()
search_tool = DuckDuckGoSearchResults(max_results=3)

wikipeida_seach_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

@tool
def get_stock_analytics(ticker: str, benchmark_selected: str = "SPY", riskfree_rate: float = 0.02) -> dict:
    """
    Retrieves P/E Ratio, Beta, and calculates Alpha (relative to S&P 500) for a given ticker.
    Also returns Market Cap and Dividend Yield.
    inputs: - ticker: stock ticker symbol (e.g., AAPL)
            - benchmark_selected: the index to compare against for alpha calculation (default is SPY)
            - riskfree_rate: return from longer term government bonds to use in alpha calculation.
    output: dict with keys 'symbol', 'pe_ratio', 'beta', 'alpha_1y', 'market_cap', 'dividend_yield', 'current_price'
    """

    ticker = ticker.replace('.', '-')  # yfinance uses '-' for tickers like BRK.B

    # to write the stock price to a local file
    if os.path.exists("stock_close_prices.csv"):
        stock_close_prices = pd.read_csv("stock_close_prices.csv", index_col=0, parse_dates=True)
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
            beta = covarience/market_varience

            # 2. Calculate returns using the same starting and ending dates
            returns = (data.iloc[-1] / data.iloc[0]) - 1
            stock_return = returns[ticker]
            market_return = returns[benchmark_selected]
            
            # 3. Standard Alpha Formula
            alpha = stock_return - (riskfree_rate + beta * (market_return - riskfree_rate))
        else:
            return{
                "error": "not enough data points to calculate alpha and beta"
            }
        
        return {
            "symbol": ticker.upper(),
            "pe_ratio": pe_ratio,
            "beta": beta,
            "alpha_1y": alpha if isinstance(alpha, float) else "N/A",
            "market_cap": market_cap if isinstance(market_cap, int) else "N/A",
            "dividend_yield": dividend_yield,
            "current_price": info.get("currentPrice", "N/A")
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