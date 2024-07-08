import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_core.runnables.history import RunnableWithMessageHistory
from uuid import uuid4

SESSION_ID = str(uuid4())
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

def getNeo4jConnection():
    db = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="ds123123",
    )
    return db


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=getNeo4jConnection())



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
    while True:
        question = input("> ")
        response = chat_with_message_history.invoke(
            {
                "context" : defeated_bosses,
                "question": question
            },
            config={"configurable": {"session_id": SESSION_ID}}
        )
        
        print(response)