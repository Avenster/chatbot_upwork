from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# 1️⃣ Load embedding model
model = SentenceTransformer("BAAI/bge-m3")

# 2️⃣ Example documents
docs = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan.",
    "The Eiffel Tower is located in Paris."
]

# 3️⃣ Create document embeddings
doc_embeddings = model.encode(docs, normalize_embeddings=True)

# 4️⃣ Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

# 5️⃣ User query
query = "What is the capital of France?"

# 6️⃣ Embed the query
query_embedding = model.encode([query], normalize_embeddings=True)

# 7️⃣ Retrieve top-k matches
k = 2
scores, indices = index.search(query_embedding, k)
top_docs = [docs[i] for i in indices[0]]

print("\n🔍 Retrieved Documents:")
for doc, score in zip(top_docs, scores[0]):
    print(f"- {doc} (Score: {score:.4f})")

# 8️⃣ Combine context + query for Groq
context = "\n".join(top_docs)
prompt = f"""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""

# 9️⃣ Generate answer using Groq
client = Groq()

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",  # Groq-hosted open-source model
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_completion_tokens=512,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None
)

print("\n🧠 Answer:\n")
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
