import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Load API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY", "")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "sample-index"

# Create the index only if it doesn't already exist
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Connect LangChain to the Pinecone index
vector_store = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings()
)

# Sample documents with metadata
docs = [
    Document(
        page_content="Virat Kohli is a top run-scorer in IPL history.",
        metadata={"team": "RCB"},
    ),
    Document(
        page_content="Rohit Sharma has led Mumbai Indians to five IPL titles.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="MS Dhoni is known as Captain Cool and led CSK to titles.",
        metadata={"team": "Chennai Super Kings"},
    ),
    Document(
        page_content="Jasprit Bumrah is a top fast bowler known for yorkers.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder for CSK.",
        metadata={"team": "Chennai Super Kings"},
    ),
]

# Store documents as vectors
ids = vector_store.add_documents(docs)
print("Added IDs:", ids)

# Semantic similarity search
results = vector_store.similarity_search(query="Who is a bowler?", k=2)
for r in results:
    print(r.page_content, "|", r.metadata)

# Similarity search with relevance scores
results_with_scores = vector_store.similarity_search_with_score(
    query="Who is a bowler?",
    k=2,
)
for doc, score in results_with_scores:
    print(f"{score:.4f} | {doc.page_content}")

# Search only within a specific team
filtered = vector_store.similarity_search(
    query="",
    k=5,
    filter={"team": "Chennai Super Kings"},
)
for r in filtered:
    print(r.page_content, "|", r.metadata)

# Remove a document by its ID
vector_store.delete(ids=[ids[0]])
print("Deleted:", ids[0])