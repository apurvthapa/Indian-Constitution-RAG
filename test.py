from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base")
model.save("./models/bge-reranker")