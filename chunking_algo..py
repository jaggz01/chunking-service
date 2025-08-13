import logging

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import torch

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print (f"using Device {DEVICE}")



# Load an open-source embedding model (small but effective)
model = SentenceTransformer("D:\\dev\\Qwen3-Embedding-0.6B", DEVICE)

# Rule-based sentence tokenizer
def simple_sent_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# Semantic chunking function
def semantic_chunk(text, similarity_threshold=0.75, max_chunk_size=5):
    sentences = simple_sent_tokenize(text)

    if not sentences:
        return []

    #Calling model for meaningful chunk creation and embeddings return
    embeddings = model.encode(sentences,
                              device=DEVICE,       #RUN on GPU
                              convert_to_tensor=True)  # Keep outputs as torch Tensors on GPU)

    #final embedding array that is getting returned.
    embeds = []
    chunks = []

    current_chunk, current_vecs = [sentences[0]], [embeddings[0]]

    for i in range(1, len(sentences)):

        similarity = cosine_similarity(
            [np.mean(current_vecs, axis=0)],
            [embeddings[i]]
        )[0][0]

        if similarity > similarity_threshold and len(current_chunk) < max_chunk_size:
            current_chunk.append(sentences[i])
            current_vecs.append(embeddings[i])
        else:
            logging.info("Chunk finalised = " , current_chunk)

            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            #finalized chunk add into the embedding array
            chunk_embeddings = model.encode(chunk_text,
                                            device=DEVICE,       #RUN on GPU
                                            convert_to_tensor=True)  # Keep outputs as torch Tensors on GPU
            embeds.append(chunk_embeddings.tolist())

            #reset or poll over
            current_chunk = [sentences[i]]
            current_vecs = [embeddings[i]]

    #finalizing the last piece of chunk that got calculated but might have not got added back because of similarity threshold check.
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_embeddings = model.encode(chunk_text,
                            device=DEVICE,       #RUN on GPU
                            convert_to_tensor=puipTrue)  # Keep outputs as torch Tensors on GPU
        embeds.append(chunk_embeddings.tolist())
        chunks.append(chunk_text)

    return chunks, embeds

# REST API endpoint
@app.route("/semantic-chunk", methods=["POST"])
def chunk_handler():
    data = request.get_json()
    text = data.get("text", "")

    #Not required to switch the context as of now
    #type = request.args.get("type", "semantic")

    threshold = float(data.get("similarity_threshold", 0.90))
    max_chunk_size = int(data.get("max_chunk_size", 5))

    logging.log(1, text, None, None);

    if not text.strip():
        return jsonify({"error": "Empty text"}), 400

    chunks, embeds = semantic_chunk(text, threshold, max_chunk_size)

    #if type == "semantic":
    #     chunks, embeds = semantic_chunk(text, threshold, max_chunk_size)
    # else:
    #    chunks, embeds = simple_sent_tokenize(text) #not tested with embeds

    return jsonify({
        "chunks": chunks,
        "embeddings": embeds,
        "chunk_count": len(chunks)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
