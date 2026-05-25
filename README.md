## Status badge
[![Python application test with GitHub Actions](https://github.com/aneeshcheriank/portfolio-builder/actions/workflows/makefile.yaml/badge.svg)](https://github.com/aneeshcheriank/portfolio-builder/actions/workflows/makefile.yaml)

---

# Portfolio Builder 📈💼

Portfolio Builder is an AI-powered multi-agent automated investment orchestration pipeline built with **LangGraph** and **LangChain**. It analyzes a user's risk tolerance, financial goals, and investable sum, identifies a benchmark baseline index, extracts top-performing stock equities via quantitative metrics, and generates an optimized asset allocation using **SciPy's SLSQP mathematical optimization solver**.

---

## 🏗️ Architecture & Agentic Workflow

The application leverages a stateful, iterative multi-agent pipeline designed using a multi-node Graph topology:

1. **Index Matcher Agent:** Interprets human user intent profiles (e.g., "$10,000, Low Risk Profile") into a specific Target Volatility metric. It loops against mathematical indices (using a custom `get_best_index_for_volatility` tool) to find the absolute closest historical index match (e.g., `SPY`, `AGG`, `QQQ`).
2. **Stock Picker Agent:** Scrapes real-time constituents of the selected index, downloads historical multi-tier equity data via Yahoo Finance (`yfinance`), computes Alpha, Beta, and P/E ratios, and down-selects the top alpha-generating stocks.
3. **Portfolio Optimizer Agent:** Feeds daily returns matrices into a custom **Markowitz Mean-Variance Optimization engine** using `scipy.optimize.minimize`. It runs Sequential Least Squares Programming (SLSQP) bounded with constraints (Min 2%, Max 20% allocation per asset; aggregate sector thresholds < 35%) to find the absolute optimized risk-adjusted weighting configuration.
4. **Structured Formatters:** Leverages Pydantic output schemas to validate data integrity across agent boundaries, exporting data models into rigid `IndexReport`, `StockSelectionReport`, and `PortfolioReport` structures.

---

## 🛠️ Tech Stack

* **Orchestration / Framework:** LangGraph, LangChain Core
* **LLM Engine:** DeepSeek V4 Flash (`ChatDeepSeek`) with structured outputs
* **Data & Analytics:** Pandas, NumPy, YFinance (`yfinance`)
* **Optimization & Math:** SciPy (`scipy.optimize`)
* **Data Validation:** Pydantic v2
* **Testing Suite:** PyTest with native mocking frameworks

---

## 📁 Repository Structure

```text
portfolio-builder/
│
├── src/
│   ├── __init__.py
│   ├── agents.py       # LangGraph agents, tool call routines, and edge routers
│   ├── chain.py        # Graph compilation topology setup (START -> Nodes -> END)
│   ├── config.py       # Global environment constants (MAX_TOOL_CALLS, etc.)
│   ├── config_env.py   # Secure dotenv loading lifecycle handlers
│   ├── model.py        # ChatDeepSeek client runtime configuration
│   ├── prompts.py      # Core System instructions and message formatting templates
│   ├── schema.py       # Pydantic data models for downstream structured outputs
│   └── tools.py        # Quantitative finance tools & optimization functions
│
└── tests/
    ├── test_chain.py   # Topology routing tests
    ├── test_schema.py  # Pydantic serialization validations
    ├── test_tools.py   # SciPy optimization & yfinance extraction unit mocks
    └── test_agents.py  # LangChain pipe (|) operator isolation and node state tests

```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python **3.11** or **3.13** installed on your machine.

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/portfolio-builder.git
cd portfolio-builder

```

### 3. Create a Virtual Environment & Install Dependencies

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate
```

# Install dependencies
`pip install -r requirements.txt`


### 4. Environment Variables Configuration

Create a `.env` file in the root directory of your project:

```env
DEEPSEEK=your_deepseek_api_key_here

```

*(Note: Ensure your exact model configuration in `src/model.py` points to the correct environment variable string defined in `src/config.py`).*

---

## 🧪 Running the Test Suite

The project includes an isolated unit-testing stack that avoids live API billings or flaky external connections by mocking `yfinance` DataFrames (using proper multi-index headers), custom SciPy convergence paths, and LangChain's internal pipeline operators (`|`).

To execute the test suite, simply execute:

```bash
pytest -v

```

### What is covered in tests?

* **`test_tools.py`:** Verifies annualized volatility derivations, Wikipedia constituent scraping, Multi-Index data alignment, and SciPy optimization boundary sums.
* **`test_agents.py`:** Mocks `get_llm()` chains using custom magic overrides on the `__or__` method to safely isolate graph nodes, verifying state modification under error boundaries.

---

## ⚙️ Core Optimization Mechanics

The `optimize_portfolio_weights` engine handles portfolio allocation mathematically:

* **Objective Function:** Minimizes portfolio variance:

$$\text{Variance} = w^T \cdot \Sigma \cdot w$$



*(Where $w$ is the array of weights and $\Sigma$ is the historical covariance matrix).*
* **Bounds:** Every individual selected asset is constrained strictly between $2\%$ and $20\%$ ($0.02 \le w_i \le 0.20$).
* **Sector Constraints:** To prevent extreme concentration risks, the total aggregate allocation targeting any single market sector (e.g., Technology, Finance) is capped strictly at $35\%$.
* **Fallback Strategy:** If historical data is highly singular or multi-collinear causing optimization failures, the pipeline catches errors and seamlessly provisions an equal-weighted portfolio fallback layout.
