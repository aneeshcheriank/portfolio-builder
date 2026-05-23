import numpy as np
import pandas as pd
from scipy.optimize import minimize
# from scipy.optimize import differential_evolution
from langchain_core.tools import tool
import yfinance as yf

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

    return {
        "sector_mapping": ticker_map,
        "daily_returns": daily_returns
    }

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
            return 500*(portfolio_vol - target_vol)**2 - weighted_returns

        # --- Constraints & Bounds ---

        # 1. Full Investment: All weights must sum to 100%
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

        # 2. Sector Constraints: Cap at 35% to allow solver flexibility
        # This prevents the solver from defaulting to equal weights
        sectors = set(sector_map.values())
        for sector in sectors:
            indices = [tickers.index(t) for t in tickers if sector_map.get(t) == sector]
            if indices:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda x, idx=indices: 0.35 - np.sum(x[idx])
                })

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
            constraints=constraints
        )

        if not result.success:
            return {"error": f"Optimization failed: {result.message}"}

        # Format output: {Ticker: Weight} rounded to 4 decimal places
        final_weights = {
            tickers[i]: round(float(result.x[i]), 4) 
            for i in range(n)
        }
        
        return {"final_weights": final_weights}

    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}
    
portfolio_optimizer_tool_mapping = {
    "optimize_portfolio_weights": optimize_portfolio_weights
}

portfolio_optimizer_tool_list = list(portfolio_optimizer_tool_mapping.values())