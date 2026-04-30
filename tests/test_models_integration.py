import os
from pathlib import Path

import pytest

from gallery.broker.pubsub import RedisBroker
from gallery.services.vector_db import VectorDBService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_model_embeds_real_image():
    """Calls the local embedding model with a real sample image."""
    if not os.environ.get("GALLERY_RUN_EMBEDDING_INTEGRATION"):
        pytest.skip("Set GALLERY_RUN_EMBEDDING_INTEGRATION=1 to run local embedding integration")
    pytest.importorskip("PIL")
    pytest.importorskip("sentence_transformers")

    sample_image = Path("samples/tomato.png").resolve()
    assert sample_image.is_file(), "Missing integration sample image: samples/tomato.png"

    service = VectorDBService(RedisBroker())
    embedding = await service._embed_image(str(sample_image))

    assert embedding, "Expected an embedding from the local model"
    assert all(isinstance(value, float) for value in embedding)
