import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
coll = client.get_or_create_collection("applicant_chunks_embedded")
print("Vector count:", coll.count())

sample = coll.get(limit=3)
for i in range(len(sample["ids"])):
    print("\nID:", sample["ids"][i])
    print("Metadata:", sample["metadatas"][i])
    print("Snippet:", sample["documents"][i][:200])