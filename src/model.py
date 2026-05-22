"""
configure the model for agents
"""
import os
from langchain_deepseek import ChatDeepSeek

from src.config_env import config_env
from src.config import api_key

config_env()

def get_llm():
    config_env()

    return ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0.0,
        api_key=os.environ.get("api_key"),
        extra_body={"thinking": { # to disable the reasoning (creates problems in formating tool)
            "type": "disabled"
        }} 
    )