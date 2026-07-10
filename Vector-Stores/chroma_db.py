"""
LangChain + Chroma Vector Store

Demonstrates:
1. Create a Chroma vector store
2. Add documents
3. Retrieve stored data
4. Perform similarity search
5. Filter by metadata
6. Update a document
7. Delete a document

Install:
    pip install langchain langchain-openai langchain-chroma langchain-community chromadb
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# OpenAI is used to convert text into embeddings.
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError(
        "OPENAI_API_KEY not set. Run `export OPENAI_API_KEY='your-key'` first."
    )

# Sample documents with metadata
docs = [
    Document(
        page_content=(
            "Virat Kohli is one of the most successful and consistent batsmen "
            "in IPL history. Known for his aggressive batting style and "
            "fitness, he has led the Royal Challengers Bangalore in multiple "
            "seasons."
        ),
        metadata={"team": "Royal Challengers Bangalore"},
    ),
    Document(
        page_content=(
            "Rohit Sharma is the most successful captain in IPL history, "
            "leading Mumbai Indians to five titles. He's known for his calm "
            "demeanor and ability to play big innings under pressure."
        ),
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content=(
            "MS Dhoni, famously known as Captain Cool, has led Chennai Super "
            "Kings to multiple IPL titles. His finishing skills, "
            "wicketkeeping, and leadership are legendary."
        ),
        metadata={"team": "Chennai Super Kings"},
    ),
    Document(
        page_content=(
            "Jasprit Bumrah is considered one of the best fast bowlers in "
            "T20 cricket. Playing for Mumbai Indians, he is known for his "
            "yorkers and death-over expertise."
        ),
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content=(
            "Ravindra Jadeja is a dynamic all-rounder who contributes with "
            "both bat and ball. Representing Chennai Super Kings, his quick "
            "fielding and match-winning performances make him a key player."
        ),
        metadata={"team": "Chennai Super Kings"},
    ),
]

# Chroma stores embeddings locally in the specified directory.
vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="my_chroma_db",
    collection_name="sample",
)

# Documents are embedded automatically before being stored.
added_ids = vector_store.add_documents(docs)
print("Added document IDs:", added_ids)

# View stored documents and their metadata.
print("\n--- Current store contents ---")
print(vector_store.get(include=["embeddings", "documents", "metadatas"]))

# Semantic search finds documents by meaning, not exact keywords.
print("\n--- Similarity search ---")
results = vector_store.similarity_search(
    query="Who among these are a bowler?",
    k=2,
)

for r in results:
    print(r.page_content, "|", r.metadata)

# Also return similarity scores.
print("\n--- Similarity search with scores ---")
results_with_scores = vector_store.similarity_search_with_score(
    query="Who among these are a bowler?",
    k=2,
)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:60]}...")

# Filter results using metadata.
print("\n--- Filter by team ---")
filtered_results = vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"},
)

for doc, score in filtered_results:
    print(f"Score: {score:.4f} | {doc.page_content[:60]}...")

# Update an existing document using its ID.
kohli_id = added_ids[0]

updated_doc = Document(
    page_content=(
        "Virat Kohli, the former captain of Royal Challengers Bangalore "
        "(RCB), is renowned for his aggressive leadership and consistent "
        "batting performances. He holds the record for the most runs in "
        "IPL history, including multiple centuries in a single season. "
        "Despite RCB not winning an IPL title under his captaincy, Kohli's "
        "passion and fitness set a benchmark for the league."
    ),
    metadata={"team": "Royal Challengers Bangalore"},
)

vector_store.update_document(document_id=kohli_id, document=updated_doc)

print("\n--- After update ---")
print(vector_store.get(include=["documents", "metadatas"]))

# Delete a document by its ID.
vector_store.delete(ids=[kohli_id])

print("\n--- After delete ---")
print(vector_store.get(include=["documents", "metadatas"]))