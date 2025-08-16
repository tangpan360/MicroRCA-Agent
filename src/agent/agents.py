"""
存放各种agent实例的模块
"""

from autogen import ConversableAgent
from .llm_config import get_llm_config


def create_log_agent() -> ConversableAgent:
    llm_config = get_llm_config()
    
    log_agent = ConversableAgent(
        name="chatbot",
        system_message="你是一个故障分析专家，请根据提供的多模态信息，分析并返回最有可能的单一故障原因，不要包含任何其他解释或文本。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    
    return log_agent

def create_llm() -> ConversableAgent:
    llm_config = get_llm_config()

    metric_agent = ConversableAgent(
        name="chatbot",
        system_message="你是一个metric分析专家。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    
    return metric_agent