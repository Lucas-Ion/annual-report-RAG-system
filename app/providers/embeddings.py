"""The dense half of retrieval: bge-m3, running locally.

Chosen over a hosted embedding API for two reasons that both matter for a
system somebody else has to run. It needs no API key, so a reviewer can clone
the repository and search without signing up for anything. And embedding 4,500
chunks is a single batch job, which is the one workload where paying per token
buys nothing.

The specific model is not interchangeable with the usual English options.
bge-m3 accepts 8,192 tokens, where bge-large-en-v1.5 and most of its
relatives stop at 512. Chunks here run to roughly 1,500 tokens, so a 512 token
model would silently truncate the largest quarter of the index: the text is
stored, displayed, cited, and completely unsearchable past the cutoff. It is
also multilingual, which matters for European filings that quote regulation in
their original language.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cached_property

import torch
from sentence_transformers import SentenceTransformer

# 1024 values per vector, which is what chunk_vectors is declared with. A model
# swap that changes this fails loudly on the first insert rather than
# corrupting the index, but it is asserted here too so the failure names the
# actual cause.
MODEL_NAME = "BAAI/bge-m3"
DIMENSIONS = 1024

# Sized for a laptop rather than for the 5090. Embedding runs once and takes
# minutes either way, so the batch that does not risk an out of memory error on
# the machine a reviewer is using is the right one.
DEFAULT_BATCH_SIZE = 8


def preferred_device() -> str:
    """Pick the best accelerator available.

    Returns:
        "cuda", "mps" or "cpu". The RAG_EMBED_DEVICE environment variable
        overrides this, which is mostly useful for forcing CPU when a GPU is
        busy doing something else.
    """
    override = os.environ.get("RAG_EMBED_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEEmbeddings:
    """bge-m3 behind the EmbeddingProvider port."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Configure the provider without loading anything.

        Construction is deliberately free of work. The model is roughly 2GB on
        disk and takes a while to load, and plenty of code paths construct a
        provider and never embed anything, the web app on startup among them.

        Args:
            model_name: Hugging Face model id.
            device: Torch device, or None to detect one.
            batch_size: Texts per forward pass.
        """
        self._model_name = model_name
        self._device = device or preferred_device()
        self._batch_size = batch_size

    @cached_property
    def _model(self) -> SentenceTransformer:
        """Load the model on first use and keep it.

        Downloads roughly 2GB from Hugging Face the first time on any given
        machine, then reads from the local cache.

        Returns:
            The loaded model.

        Raises:
            RuntimeError: If the model's width is not what chunk_vectors was
                declared with, which would otherwise surface as an opaque
                error from SQLite several stages later.
        """
        model = SentenceTransformer(self._model_name, device=self._device)

        # Renamed in sentence-transformers 5. Both names are tried so the
        # version pin can stay loose without a deprecation warning on load.
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
        """Vector width, without loading the model.

        Returns:
            The declared dimension count.
        """
        return DIMENSIONS

    @property
    def device(self) -> str:
        """Which device this provider will use.

        Returns:
            The torch device name.
        """
        return self._device

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunks for storage.

        Vectors are normalised to unit length, which is what makes cosine
        similarity and plain dot product equivalent. sqlite-vec measures L2
        distance, and on normalised vectors that ranks identically to cosine,
        so this is what keeps the stored index consistent with how retrieval
        expects distances to behave.

        Args:
            texts: Chunk texts, each already carrying its context header.

        Returns:
            One vector per input, in the same order. An empty input gives an
            empty list rather than an error, since a document with nothing to
            embed is an ordinary thing during a resumed run.
        """
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
        """Embed a question for searching.

        bge-m3 wants no instruction prefix on either side, so this is the same
        call as embed_documents. It stays a separate method because that is
        not true of most alternatives, and a model swap should not be able to
        quietly introduce an asymmetry bug.

        Args:
            text: The user's question.

        Returns:
            One vector.
        """
        return self.embed_documents([text])[0]
