import os
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

docs = [
    Document(page_content="Virat Kohli is a top run-scorer in IPL history.", metadata={"team": "RCB"}),
    Document(page_content="Rohit Sharma has led Mumbai Indians to five IPL titles.", metadata={"team": "Mumbai Indians"}),
    Document(page_content="MS Dhoni is known as Captain Cool and led CSK to titles.", metadata={"team": "Chennai Super Kings"}),
    Document(page_content="Jasprit Bumrah is a top fast bowler known for yorkers.", metadata={"team": "Mumbai Indians"}),
    Document(page_content="Ravindra Jadeja is a dynamic all-rounder for CSK.", metadata={"team": "Chennai Super Kings"}),
]

vector_store = Milvus(
    embedding_function=OpenAIEmbeddings(),
    collection_name="sample",
    connection_args={"uri": "./milvus_demo.db"},  # local Milvus Lite file
    auto_id=True,
)

ids = vector_store.add_documents(docs)
print("Added IDs:", ids)

results = vector_store.similarity_search(query="Who is a bowler?", k=2)
for r in results:
    print(r.page_content, "|", r.metadata)

results_with_scores = vector_store.similarity_search_with_score(query="Who is a bowler?", k=2)
for doc, score in results_with_scores:
    print(f"{score:.4f} | {doc.page_content}")

filtered = vector_store.similarity_search(query="", k=5, expr='team == "Chennai Super Kings"')
for r in filtered:
    print(r.page_content, "|", r.metadata)

vector_store.delete(ids=[ids[0]])
print("Deleted:", ids[0])