import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

def call_openai_api():
    """
    Calls the OpenAI API with the given prompt and returns the response.

    :param prompt: The prompt to send to the OpenAI API.
    :return: The response from the OpenAI API.
    """
    # Retrieve the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    # Create an instance of the OpenAI class with your API key
    llm = OpenAI(
        openai_api_key=api_key,
         model="gpt-3.5-turbo-instruct",
        temperature=0
    )

    return llm

# Example usage
if __name__ == "__main__":
    template = PromptTemplate(template="""You are a Elden Ring expert and you know all about Fromsoftware game Elden Ring. 
                              I want to defeat the boss {boss}, what is the best equipement, magics and strategy I can se to defeat him?
                              take in consideration I'm creating a character using the following Attributes: {attributes}""", input_variables=["boss","attributes"])
    llm_chain = template | call_openai_api()
    response = llm_chain.invoke({"boss": "Elden Beast", "attributes": "Strength"})
    print(response)
