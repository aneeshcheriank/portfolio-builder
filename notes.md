## Structure of the config (InMemorySaver)

```python
config = {
    # 1. Standard LangGraph Fields
    "configurable": {
        "thread_id": "session_12345",  # Unique ID for the conversation checkpoint
        "checkpoint_id": "1ef23-abc...", # (Optional) Points to a specific turn in history
    },
    
    # 2. Custom Metadata Fields (Optional)
    "metadata": {
        "user_id": "user_alice",
        "app_version": "v2.1.0"
    },
    
    # 3. Execution Tags (Optional)
    "tags": ["production", "web_chat"]
}
```

### configurable
- thread_id: the memory location to look to get the converation historz
y
- checkpoint_id: more like the time in the converation history


## Human-feedback only one time in the loop
```python
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
```
- `stream_mode`
    - what kind of data emitted and how oftern it is pushed out when graph executed
    - 3 most common streaming mode
        - "updates"
            - information comming out everytime a node complete its execution
            - it shows the dictionaries which the node changes
            - uses
                - progress tracking
                - real time node execution
                - lightweight logging
        - "values"
            - insted of edits, shows the state after every single step
            - uses
                - debugging exactly what the sate looks like at any given millisecond
        - "messeges"
            - If your agent is talking to a user and you want a ChatGPT-like experience where words appear on the screen bit-by-bit (token streaming), you use "messages".
            - uses
                - Streaming raw text/tokens to a frontend chat interface.

## Infinite loop
```python
# main.py
from src.chain import build_graph
from langgraph.errors import GraphInterrupt # If your version uses it, otherwise check state

if __name__ == "__main__":
    question = """
    what is a good investment strategy for a moderate risk investor with 1000 to invest for 10 years?
    """

    workflow = build_graph()
    config = {"configurable": {"thread_id": "thread-1"}}

    # Initialize the graph
    initial_state = {
        "user_input": question,
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

    # Step 1: First kick-off
    print("--- Starting initial portfolio generation ---")
    events = workflow.stream(initial_state, config, stream_mode="updates")
    for event in events:
        print(event)

    # Step 2: Keep looping as long as the graph is interrupted before feedback_collector
    while True:
        state_snapshot = workflow.get_state(config)
        
        # Check if the graph is currently waiting at a breakpoint
        if not state_snapshot.next:
            # No next nodes means the graph finished completely (Hit END)
            print("\n=== WORKFLOW COMPLETE ===")
            break
            
        if "feedback_collector" in state_snapshot.next:
            print("\n=== SYSTEM PAUSED FOR HUMAN REVIEW ===")
            
            # Print the current explanation to the user
            current_values = state_snapshot.values
            if "explanation" in current_values:
                print(f"\nLatest Explanation: {current_values['explanation'].content}")
            
            # Take input safely
            user_feedback_input = input("\nEnter suggestions, or type 'Approved' to finish: ")
            
            # If they approve, we can set approval directly or let the node process it
            # Let's push the raw text in
            workflow.update_state(
                config,
                {"feedback": user_feedback_input},
                as_node="portfolio_explainer"
            )
            
            print("\n--- Feedback injected. Resuming execution loop... ---")
            
            # Resume stream. It will run until it hits END or loops back to this breakpoint again!
            events = workflow.stream(None, config, stream_mode="updates")
            for event in events:
                print(event)
```