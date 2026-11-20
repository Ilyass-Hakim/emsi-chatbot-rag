"""Hybrid retrieval over the Qdrant collection.

A question is searched in several ways, and the result lists are merged
with Reciprocal Rank Fusion (RRF):
  - the question against text chunks and image captions (BGE-M3);
  - the question against the images themselves (multilingual CLIP);
  - an optional photo sent by the user against the images (CLIP)."""

import io
import sys

from PIL import Image
from qdrant_client import QdrantClient, models

import config
from embeddings import clip_image_model, clip_text_model, text_model

client = QdrantClient(url=config.QDRANT_URL)


def search(question: str, k: int = 5, photo: bytes | None = None) -> list[dict]:
    """Return the k most relevant records for a question and an optional photo."""
    prefetch = [
        # The question against text chunks and image captions
        models.Prefetch(
            query=text_model().encode(question, normalize_embeddings=True).tolist(),
            using="text", limit=20,
        ),
        # The question against the images themselves
        models.Prefetch(
            query=clip_text_model().encode(question).tolist(),
            using="image", limit=10,
        ),
    ]
    # Text matches count fully; image matches from the question count less,
    # so that unrelated pictures do not push relevant text out of the top k.
    weights = [1.0, config.IMAGE_SEARCH_WEIGHT]

    if photo:
        image = Image.open(io.BytesIO(photo)).convert("RGB")
        prefetch.append(models.Prefetch(
            query=clip_image_model().encode(image).tolist(), using="image", limit=10,
        ))
        weights.append(1.0)

    response = client.query_points(
        collection_name=config.COLLECTION,
        prefetch=prefetch,
        query=models.RrfQuery(rrf=models.Rrf(k=config.RRF_K, weights=weights)),
        limit=k,
        with_payload=True,
    )
    return [{**point.payload, "score": point.score} for point in response.points]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Depuis quand existe l'EMSI ?"
    print(f"Question: {question}\n")
    for rank, hit in enumerate(search(question), start=1):
        print(f"{rank}. page {hit['page']:>2} | {hit['type']:<5} | score {hit['score']:.4f}")
        print(f"   {hit['content'][:150]}\n")
