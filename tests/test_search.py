"""
Unit tests for visual search engine.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.search import VisualSearchEngine


class TestVisualSearchEngine:
    """Test visual search functionality."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        np.random.seed(42)
        return np.random.randn(100, 512).astype('float32')

    @pytest.fixture
    def sample_product_ids(self):
        """Create sample product IDs."""
        return [f"PROD_{i:04d}" for i in range(100)]

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return pd.DataFrame({
            'id': [f"PROD_{i:04d}" for i in range(100)],
            'name': [f"Product {i}" for i in range(100)],
            'category': np.random.choice(['Dress', 'Shirt', 'Jeans'], 100),
            'price': np.random.uniform(20, 200, 100)
        })

    @pytest.fixture
    def search_engine(self, sample_embeddings, sample_product_ids, sample_metadata):
        """Create search engine with sample data."""
        engine = VisualSearchEngine(embedding_dim=512)
        engine.build_index(
            embeddings=sample_embeddings,
            product_ids=sample_product_ids,
            metadata=sample_metadata
        )
        return engine

    def test_initialization(self):
        """Test search engine initialization."""
        engine = VisualSearchEngine(embedding_dim=512)
        assert engine.embedding_dim == 512
        assert engine.index is None

    def test_build_index(self, sample_embeddings, sample_product_ids):
        """Test index building."""
        engine = VisualSearchEngine(embedding_dim=512)
        engine.build_index(
            embeddings=sample_embeddings,
            product_ids=sample_product_ids
        )

        assert engine.index is not None
        assert engine.index.ntotal == 100

    def test_search(self, search_engine, sample_embeddings):
        """Test basic search."""
        query = sample_embeddings[0]
        results = search_engine.search(query, top_k=5)

        assert len(results) == 5
        assert all('product_id' in r for r in results)
        assert all('similarity' in r for r in results)

        # First result should be highest similarity
        similarities = [r['similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_search_with_threshold(self, search_engine, sample_embeddings):
        """Test search with similarity threshold."""
        query = sample_embeddings[0]
        results = search_engine.search(query, top_k=10, threshold=0.9)

        # Results should meet threshold
        assert all(r['similarity'] >= 0.9 for r in results)

    def test_search_with_filters(self, search_engine, sample_embeddings):
        """Test search with filters."""
        query = sample_embeddings[0]
        results = search_engine.search_with_filters(
            query,
            top_k=5,
            filters={'category': 'Dress'}
        )

        # All results should match filter
        assert all(r['category'] == 'Dress' for r in results)

    def test_save_load_index(self, search_engine, tmp_path):
        """Test saving and loading index."""
        index_path = tmp_path / "test_index.faiss"

        # Save
        search_engine.save_index(str(index_path))
        assert index_path.exists()

        # Load into new engine
        new_engine = VisualSearchEngine(embedding_dim=512)
        new_engine.load_index(str(index_path))

        assert new_engine.index.ntotal == 100
        assert len(new_engine.product_ids) == 100

    def test_get_statistics(self, search_engine):
        """Test statistics retrieval."""
        stats = search_engine.get_statistics()

        assert stats['total_vectors'] == 100
        assert stats['embedding_dim'] == 512
        assert stats['total_products'] == 100
