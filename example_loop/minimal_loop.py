from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()  # in-memory store
col = client.create_collection("minimal")

# Step 2: embed and store one text
t1 = "Employees may work remotely up to three days per week with manager approval."
col.add(ids=["doc1"], documents=[t1], embeddings=[model.encode(t1).tolist()])
print("Stored 1 chunk. Count:", col.count())

# Step 3: add a second, different text
t2 = "Meal expenses during business travel are reimbursed up to $50 per day."
col.add(ids=["doc2"], documents=[t2], embeddings=[model.encode(t2).tolist()])
print("Stored 2 chunks. Count:", col.count())

# Step 4L add a third, different text
t3 = "Employees may not work on weekends unless they are on call."
col.add(ids=["doc3"], documents=[t3], embeddings=[model.encode(t3).tolist()])
print("Stored 3 chunks. Count:", col.count())

# Query and confirm the more relevant one comes back first
q = "How many days can I work from home?"
res = col.query(query_embeddings=[model.encode(q).tolist()], n_results=3)
print("\nQuery:", q)
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"  distance={dist:.4f}  {doc}")
assert res["ids"][0][0] == "doc1", "Wrong chunk ranked first!"
print("\nPASS: remote work text ranked first.")