import os
from pathlib import Path

import pytest

from gallery.broker.pubsub import RedisBroker
from gallery.services.vector_db import VectorDBService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_github_models_annotates_real_image():
    """Calls GitHub Models with a real sample image and checks annotations."""
    if not os.environ.get("MODELS_TOKEN"):
        pytest.skip("MODELS_TOKEN is not set")

    sample_image = Path("samples/tomato.png").resolve()
    assert sample_image.is_file(), "Missing integration sample image: samples/tomato.png"

    service = VectorDBService(RedisBroker())
    annotations = await service._annotate_image(str(sample_image))

    assert annotations, "Expected at least one annotation from GitHub Models"
    assert len(annotations) <= 5
    assert all(isinstance(tag, str) and tag.strip() for tag in annotations)
