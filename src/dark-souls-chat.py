import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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

# Example usage
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
    ])
    chat_llm = get_openAPI_llm()
    chat_chain = prompt | chat_llm | StrOutputParser()
    defeated_bosses = """
    {
        "defeated_bosses": [
            "Moonlight Butterfly",
            "Taurus Demon",
        ]
    }
    """
    response = chat_chain.invoke({
        "context" : defeated_bosses,
        "question": "What is the next easier boss for me to fight? Take in consideration the list of defeated bosses I already killed"
        })
    print(response)