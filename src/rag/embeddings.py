from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(config.EMBED_MODEL)


def embed_passages(texts):
    vecs = _model().encode([f"passage: {t}" for t in texts], normalize_embeddings=True)
    return np.asarray(vecs)


def embed_query(text):
    return _model().encode([f"query: {text}"], normalize_embeddings=True)[0]
