from src.chain import build_graph

if __name__ == "__main__":

    question = """
    what is a good investmetn stragety for a moderate risk investor with 1000 to invest for 10 years?
    """

    workflow = build_graph()
    # a user identifier
    config = {"configurable": {"thread_id": "thread-1"}}

    initial_state = {
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
        }
    
    print("-----------Executing Grpah-----------")
    # use stream insted of invoke (this will allow to catch the breakpoint in the feedback collector node)
    for event in workflow.stream(initial_state, config=config, stream_mode="updates"):
        print(event)

    # graph pause before the "feedback_collector" node
    print("--------Graph Paused for Feedback--------")
    print("The portfolio explanation has beed generated.")

    user_feedback = input("Do you have any suggestion or feedback on the preposed portfolio? (press enter to approve)")

    # update the feedback in the state
    workflow.update_state(
        config,
        {"feedback": user_feedback},
        as_node="portfolio_explainer" # the last executed node before the pause
    )

    print("--------Resuming Graph Execution--------")
    # resume the graph execution after getting the feedback
    final_output = None
    for event in workflow.stream(None, config, stream_mode="updates"):
        print(event)
        final_ouput = event

    # ouputs
    final_state = workflow.get_state(config).values
    print("\n===========Final Result===========")
    print("final portfolio:", final_state.get("portfolio"))