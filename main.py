from langchain_openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key="api_key")

response = llm.invoke("What is Neo4j?")

print(response)