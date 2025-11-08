"""
Unit tests for CLIP embeddings module.
"""

import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.embeddings import CLIPEmbedder


class TestCLIPEmbedder:
    """Test CLIP embedding functionality."""

    @pytest.fixture
    def embedder(self):
        """Create CLIP embedder instance."""
        return CLIPEmbedder(model_name="ViT-B/32")

    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return Image.new('RGB', (224, 224), color='red')

    def test_initialization(self, embedder):
        """Test embedder initialization."""
        assert embedder is not None
        assert embedder.model is not None
        assert embedder.device in ['cpu', 'cuda']

    def test_encode_text(self, embedder):
        """Test text encoding."""
        text = "red dress for women"
        embedding = embedder.encode_text(text)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 512  # CLIP ViT-B/32 embedding dimension

    def test_encode_text_batch(self, embedder):
        """Test batch text encoding."""
        texts = ["red dress", "blue jeans", "black shoes"]
        embeddings = embedder.encode_text(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 512)

    def test_encode_image(self, embedder, sample_image):
        """Test image encoding."""
        embedding = embedder.encode_image(sample_image)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 512

    def test_compute_similarity(self, embedder):
        """Test similarity computation."""
        emb1 = embedder.encode_text("red dress")
        emb2 = embedder.encode_text("red dress")
        emb3 = embedder.encode_text("blue jeans")

        # Same text should have high similarity
        similarity_same = embedder.compute_similarity(emb1, emb2)
        assert 0.99 <= similarity_same <= 1.0

        # Different text should have lower similarity
        similarity_diff = embedder.compute_similarity(emb1, emb3)
        assert similarity_diff < similarity_same

    def test_multimodal_query(self, embedder, sample_image):
        """Test multimodal query creation."""
        combined = embedder.create_multimodal_query(
            text="red dress",
            image=sample_image,
            text_weight=0.5
        )

        assert isinstance(combined, np.ndarray)
        assert combined.shape == (512,)

        # Check normalization
        norm = np.linalg.norm(combined)
        assert 0.99 <= norm <= 1.01

    def test_embedding_normalization(self, embedder):
        """Test that embeddings are normalized."""
        text = "test text"
        embedding = embedder.encode_text(text)

        norm = np.linalg.norm(embedding)
        assert 0.99 <= norm <= 1.01  # Should be unit vector
