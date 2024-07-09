import os

from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

api_key = os.getenv("OPENAI_API_KEY")

embedding_provider = OpenAIEmbeddings(
    openai_api_key=api_key
)

llm = ChatOpenAI(openai_api_key=api_key)

graph = Neo4jGraph(
    url="bolt://44.220.85.60:7687",
    username="neo4j",
    password="steeple-attention-wiggles"
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever(),
    verbose=True,
    return_source_documents=True
)

result = plot_retriever.invoke(
    {"query": "A movie where a mission to the moon goes wrong"}
)
print(result)