import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


memory = ChatMessageHistory()

def get_openAPI_llm():
    # Retrieve the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    # Create an instance of the OpenAI class with your API key
    chat_llm = ChatOpenAI(
        openai_api_key=api_key
    )

    return chat_llm

def get_chat_history():
    history = Neo4jChatMessageHistory(
        url="bolt://localhost:7687",
        username="neo4j",
        password="ds123123",
        session_id="session_id_1",
    )
    return history


def get_memory(session_id):
    return memory


if __name__ == "__main__":
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a professional Dark Souls player and you know all about Fromsoftware game Dark Souls."
        ),
        (
            "human",
            "{question}"
        ),
        (
            "system",
            "{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
    ])
    chat_llm = get_openAPI_llm()
    
    chat_chain = prompt | chat_llm | StrOutputParser()
    
    chat_with_message_history = RunnableWithMessageHistory(
        chat_chain,
        get_memory,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    defeated_bosses = """
    {
        "defeated_bosses": [
            "Moonlight Butterfly",
            "Taurus Demon",
        ]
    }
    """
    session_id = "unique_session_id_1"

    while True:
        question = input("> ")
        response = chat_with_message_history.invoke(
            {
                "context" : defeated_bosses,
                "question": question
            },
            config={"configurable": {"session_id": session_id}}
        )
        
        print(response)