import os
import json

from colorama import Fore, Style, init
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools import YouTubeSearchTool
from uuid import uuid4

SESSION_ID = str(uuid4())
youtube = YouTubeSearchTool()


def getOpenAPILlm():
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

def getTools(chat_chain):
    tools = [
        Tool.from_function(
            name="Dark Souls Chat",
            description="For when you need to chat Dark Souls. The question will be a string. Return a string.",
            func=chat_chain.invoke,
        ),
        Tool.from_function(
            name="Youtube Dark Souls",
            description="Use when needing to share steps, or tutorials finding an item. The question will include the word find. Return a link to a YouTube video for DARK SOULS only.",
            func=callTrailerSearch,
        ),
    ]
    return tools

def getMemory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=getNeo4jConnection())

def callTrailerSearch(input):
    input = input.replace(",", " ") + "In Dark Souls"
    input
    return youtube.run(input)

if __name__ == "__main__":
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional Dark Souls player and you know all about Fromsoftware game Dark Souls."
            ),
            (
                "human",
                "{input}"
            ),
            (
              "system",
              "This bosses I already defeated: Moonlight Butterfly, Taurus Demon"
            ),
      ]
    )
    chat_llm = getOpenAPILlm()
    
    chat_chain = prompt | chat_llm | StrOutputParser()
    tools = getTools(chat_chain)
    agent_prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(chat_llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    chat_agent = RunnableWithMessageHistory(
        agent_executor,
        getMemory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    

    while True:
        print(Fore.WHITE)
        question = input("> ")
        response = chat_agent.invoke(
            {
                "input": question
            },
            config={"configurable": {"session_id": SESSION_ID}}
        )
        
        print(Fore.GREEN + response["output"])
