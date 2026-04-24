import json

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from jina.model import JinaEmbeddingModel

with open("jina/config.json") as f:
    config = json.load(f)

model = JinaEmbeddingModel(config)
weights = mx.load("jina/model.safetensors")
model.load_weights(list(weights.items()))

tokenizer = Tokenizer.from_file("jina/tokenizer.json")


def embed(messages: list[dict]) -> np.ndarray:
    text = "\n".join(map(lambda m: m["text"], messages))
    embeddings = model.encode([text], tokenizer, task_type="text-matching")
    mx.clear_cache()
    return np.array(embeddings[0])


if __name__ == "__main__":
    for i in range(10000):
        embed([{"text": f"{i}"}])
        print(f"\r {i}", end="")
