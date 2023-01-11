# python3 -m pip install sentence-transformers

from sentence_transformers import SentenceTransformer
#sentences = ["george Washington was president"]
sentences = ["Hello I'm a single sentence"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)