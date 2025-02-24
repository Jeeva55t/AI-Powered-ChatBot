import faiss
import numpy as np
import os

# Ensure the directory exists
os.makedirs("faiss_index", exist_ok=True)

# Define dimensions (adjust based on your use case)
d = 128  # Example: 128-dimensional vectors
index = faiss.IndexFlatL2(d)  # L2 distance index

# Generate random data for testing (replace this with your real embeddings)
vectors = np.random.random((1000, d)).astype('float32')

# Add vectors to the FAISS index
index.add(vectors)

# Save the index to a file
faiss.write_index(index, "faiss_index/index.faiss")

print("FAISS index created and saved successfully.")
