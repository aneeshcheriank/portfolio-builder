from src.chain import build_graph

if __name__ == "__main__":

    question = """
    what is a good investmetn stragety for a moderate risk investor with 1000 to invest for 10 years?
    """

    workflow = build_graph()
    # a user identifier
    config = {"configurable": {"thread_id": "thread-1"}}
    response = workflow.invoke(
        {
            "user_input": question,  # need to initialze the values in then only it will be availble to use in state variable
            "investing_sum": 0,
            "risk_class": "Low",
            "expected_return": 0.02,
            "chat_history": [],
            "stock_picker_history": [],
            "portfolio_optimizer_history": [],
            "iterations": 0,
            "iterations_stock_picker": 0,
            "iterations_portfolio_optimizer": 0,
            "risk_free_rate": 0.02,
            "feedback": "",
            "approval": False
        }, config
    )

    print(response["portfolio"])
    print(response["explanation"].content)
