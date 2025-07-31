import logging

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = Flask(__name__)

# Load an open-source embedding model (small but effective)
model = SentenceTransformer("qwen3")

# Rule-based sentence tokenizer
def simple_sent_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# Semantic chunking function
def semantic_chunk(text, similarity_threshold=0.75, max_chunk_size=5):
    sentences = simple_sent_tokenize(text)
    if not sentences:
        return []

    embeddings = model.encode(sentences)
    chunks, current_chunk, current_vecs = [], [sentences[0]], [embeddings[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            [np.mean(current_vecs, axis=0)],
            [embeddings[i]]
        )[0][0]

        if similarity > similarity_threshold and len(current_chunk) < max_chunk_size:
            current_chunk.append(sentences[i])
            current_vecs.append(embeddings[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_vecs = [embeddings[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# REST API endpoint
@app.route("/semantic-chunk", methods=["POST"])
def chunk_handler():
    data = request.get_json()
    text = data.get("text", "")
    threshold = float(data.get("similarity_threshold", 0.75))
    max_chunk_size = int(data.get("max_chunk_size", 5))

    logging.log(logging.info, text, None, None);

    if not text.strip():
        return jsonify({"error": "Empty text"}), 400

    chunks = semantic_chunk(text, threshold, max_chunk_size)
    return jsonify({"chunks": chunks, "chunk_count": len(chunks)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
