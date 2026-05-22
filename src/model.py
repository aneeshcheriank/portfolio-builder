"""
configure the model for agents
"""

# Now that the environment is loaded, it's safe to import your configuration
from src.configuration import MODEL
from src.environment import config_env

# 4. Import LangChain now that the API key is securely set in os.environ
# from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek
import os

# def get_llm():
#     config_env()

#     return ChatGroq(
#         model=MODEL,
#         temperature=0.0,
#         api_key=os.environ.get("GROQ_API_KEY")
#     )

def get_llm():
    config_env()

    return ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0.0,
        api_key=os.environ.get("DEEPSEEK"),
        extra_body={"thinking": { # to disable the reasoning (creates problems in formating tool)
            "type": "disabled"
        }} 
    )