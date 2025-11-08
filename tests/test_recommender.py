"""
Unit tests for recommendation engine.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.recommender import RecommendationEngine


class TestRecommendationEngine:
    """Test recommendation functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample product data."""
        return pd.DataFrame({
            'id': [f"PROD_{i:04d}" for i in range(50)],
            'productDisplayName': [f"Product {i}" for i in range(50)],
            'category': np.random.choice(['Dress', 'Shirt', 'Jeans', 'Shoes'], 50),
            'baseColour': np.random.choice(['Red', 'Blue', 'Black', 'White'], 50),
            'gender': np.random.choice(['Men', 'Women', 'Unisex'], 50),
            'season': np.random.choice(['Summer', 'Winter', 'Fall', 'Spring'], 50),
            'price': np.random.uniform(20, 200, 50)
        })

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        np.random.seed(42)
        return np.random.randn(50, 512).astype('float32')

    @pytest.fixture
    def recommender(self, sample_data):
        """Create recommender instance."""
        return RecommendationEngine(sample_data)

    def test_initialization(self, sample_data):
        """Test recommender initialization."""
        recommender = RecommendationEngine(sample_data)
        assert recommender.metadata is not None
        assert len(recommender.metadata) == 50

    def test_content_based_recommendations(
        self, recommender, sample_embeddings, sample_data
    ):
        """Test content-based recommendations."""
        product_id = "PROD_0000"
        product_ids = sample_data['id'].tolist()

        recs = recommender.content_based_recommendations(
            product_id,
            sample_embeddings,
            product_ids,
            top_n=5
        )

        assert len(recs) <= 5
        assert all('similarity' in r for r in recs)
        assert all('reason' in r for r in recs)

        # Source product should not be in recommendations
        assert all(r['id'] != product_id for r in recs)

    def test_attribute_based_recommendations(self, recommender):
        """Test attribute-based recommendations."""
        product_id = "PROD_0000"

        recs = recommender.attribute_based_recommendations(
            product_id,
            top_n=5
        )

        assert len(recs) <= 5
        assert all('score' in r for r in recs)
        assert all('reason' in r for r in recs)

    def test_complementary_recommendations(self, recommender):
        """Test complementary product recommendations."""
        product_id = "PROD_0000"

        recs = recommender.complementary_recommendations(
            product_id,
            top_n=5
        )

        # May return fewer if no complementary categories
        assert len(recs) <= 5

    def test_hybrid_recommendations(
        self, recommender, sample_embeddings, sample_data
    ):
        """Test hybrid recommendations."""
        product_id = "PROD_0000"
        product_ids = sample_data['id'].tolist()

        recs = recommender.hybrid_recommendations(
            product_id,
            sample_embeddings,
            product_ids,
            top_n=5
        )

        assert len(recs) <= 5
        assert all('recommendation_score' in r for r in recs)
        assert all('recommendation_sources' in r for r in recs)

        # Scores should be descending
        scores = [r['recommendation_score'] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_trending_products(self, recommender):
        """Test trending products retrieval."""
        trending = recommender.trending_products(top_n=10)

        assert len(trending) <= 10
        assert all('reason' in r for r in trending)
