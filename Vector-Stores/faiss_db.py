import os
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

docs = [
    Document(page_content="Virat Kohli is a top run-scorer in IPL history.", metadata={"team": "RCB"}),
    Document(page_content="Rohit Sharma has led Mumbai Indians to five IPL titles.", metadata={"team": "Mumbai Indians"}),
    Document(page_content="MS Dhoni is known as Captain Cool and led CSK to titles.", metadata={"team": "Chennai Super Kings"}),
    Document(page_content="Jasprit Bumrah is a top fast bowler known for yorkers.", metadata={"team": "Mumbai Indians"}),
    Document(page_content="Ravindra Jadeja is a dynamic all-rounder for CSK.", metadata={"team": "Chennai Super Kings"}),
]

embeddings = OpenAIEmbeddings()
index = faiss.IndexFlatL2(len(embeddings.embed_query("dummy")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

ids = vector_store.add_documents(docs)
print("Added IDs:", ids)

results = vector_store.similarity_search(query="Who is a bowler?", k=2)
for r in results:
    print(r.page_content, "|", r.metadata)

results_with_scores = vector_store.similarity_search_with_score(query="Who is a bowler?", k=2)
for doc, score in results_with_scores:
    print(f"{score:.4f} | {doc.page_content}")

filtered = vector_store.similarity_search(query="", k=5, filter={"team": "Chennai Super Kings"})
for r in filtered:
    print(r.page_content, "|", r.metadata)

vector_store.delete(ids=[ids[0]])
print("Deleted:", ids[0])

vector_store.save_local("faiss_index")
reloaded = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print("Reloaded doc count:", len(reloaded.docstore._dict))