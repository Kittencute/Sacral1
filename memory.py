from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOllama

def create_summary_memory(model_name: str, memory_key: str = "chat_history", token_limit: int = 1000):
    llm = ChatOllama(model=model_name)

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key=memory_key,
        return_messages=True,
        max_token_limit=token_limit
    )

    return memory
