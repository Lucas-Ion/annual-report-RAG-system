from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cached_property

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
DIMENSIONS = 1024
DEFAULT_BATCH_SIZE = 8


def preferred_device() -> str:
    override = os.environ.get("RAG_EMBED_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEEmbeddings:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._device = device or preferred_device()
        self._batch_size = batch_size

    @cached_property
    def _model(self) -> SentenceTransformer:
        model = SentenceTransformer(self._model_name, device=self._device)
        measure = getattr(model, "get_embedding_dimension", None)
        width = measure() if measure else model.get_sentence_embedding_dimension()
        if width != DIMENSIONS:
            raise RuntimeError(
                f"{self._model_name} produces {width} dimensional vectors, "
                f"but chunk_vectors is declared FLOAT[{DIMENSIONS}]. "
                f"Change the schema and re-embed, or use a different model."
            )
        return model

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    @property
    def device(self) -> str:
        return self._device

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
