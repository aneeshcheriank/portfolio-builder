## portfolio maker
- objetive
    - create a portfolio for a risk and investable sum
    - the portrfolio for Europien market, and is pssive
        - it tracks a Index (ETF) and followed the index movement for management to reduce the active management cost

### Execution
- LangGraph workflow
    - feedback loops and Mathematical optimization
- [workflow](./Readme.md#)

## Running the application
- install the required packages
- `pip install -r requirement.txt`
- from applicaton folder run terminal
- terminal: `python main.py`
- .env file
    - create a .env file in the root folder
    - update the "DEEPSEEK = xxxxxxxxxxxxx" with the API KEY

## Testing
- to run all the test
- `pytest`
- to test a specific module
- `pytest tests/test_model.py`