from pypdf import PdfReader

pdf_path = r"C:\Users\cgkis\OneDrive\Documents\HEDIS-MY-2024-Measure-Description.pdf"
reader = PdfReader(pdf_path)


# ---------- TEXT EXTRACTION ----------
texts = []
for page in reader.pages:
    t = page.extract_text()
    if t:
        texts.append(t)

full_text = "\n".join(texts)
print(full_text[:2000])

import re

measures = re.split(r"\n[A-Z][a-zA-Z ,\-()]+ [A-Z]{2,4}\n", full_text)

clean_chunks = []
for chunk in measures:
    chunk = chunk.strip()
    if len(chunk) > 100:  # avoid tiny fragments
        clean_chunks.append(chunk)

print(f"Total chunks: {len(clean_chunks)}")

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(clean_chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "hedis_index.faiss")

with open("hedis_chunks.pkl", "wb") as f:
    pickle.dump(clean_chunks, f)

print("HEDIS vector DB saved!")
