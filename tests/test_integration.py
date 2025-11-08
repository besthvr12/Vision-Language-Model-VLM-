"""
Integration tests for end-to-end workflows.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DatasetLoader
from src.models.embeddings import CLIPEmbedder
from src.models.search import VisualSearchEngine
from src.models.recommender import RecommendationEngine


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.fixture(scope="class")
    def setup_system(self):
        """Set up complete system for testing."""
        # Load data
        loader = DatasetLoader()
        df = loader.create_sample_dataset(num_samples=50)

        # Initialize embedder
        embedder = CLIPEmbedder()

        # Generate embeddings
        descriptions = df.apply(
            lambda x: f"{x['category']} {x['baseColour']} for {x['gender']}",
            axis=1
        ).tolist()
        embeddings = embedder.encode_text(descriptions)

        # Build search engine
        search_engine = VisualSearchEngine(embedding_dim=embeddings.shape[1])
        search_engine.build_index(
            embeddings=embeddings,
            product_ids=df['id'].tolist(),
            metadata=df
        )

        # Initialize recommender
        recommender = RecommendationEngine(df)

        return {
            'df': df,
            'embedder': embedder,
            'embeddings': embeddings,
            'search_engine': search_engine,
            'recommender': recommender
        }

    def test_search_workflow(self, setup_system):
        """Test complete search workflow."""
        embedder = setup_system['embedder']
        search_engine = setup_system['search_engine']

        # User query
        query = "red dress for women"

        # Generate query embedding
        query_embedding = embedder.encode_text(query)

        # Search
        results = search_engine.search(query_embedding, top_k=5)

        # Verify results
        assert len(results) == 5
        assert all('product_id' in r for r in results)
        assert all('similarity' in r for r in results)
        assert all('price' in r for r in results)

    def test_recommendation_workflow(self, setup_system):
        """Test complete recommendation workflow."""
        df = setup_system['df']
        embeddings = setup_system['embeddings']
        recommender = setup_system['recommender']

        # Select a product
        product_id = df.iloc[0]['id']

        # Get recommendations
        recs = recommender.hybrid_recommendations(
            product_id,
            embeddings,
            df['id'].tolist(),
            top_n=5
        )

        # Verify recommendations
        assert len(recs) <= 5
        assert all('id' in r for r in recs)
        assert all('recommendation_score' in r for r in recs)

        # Original product should not be in recommendations
        assert all(r['id'] != product_id for r in recs)

    def test_search_and_recommend_workflow(self, setup_system):
        """Test search followed by recommendations."""
        embedder = setup_system['embedder']
        search_engine = setup_system['search_engine']
        recommender = setup_system['recommender']
        df = setup_system['df']
        embeddings = setup_system['embeddings']

        # Step 1: Search
        query = "blue jeans"
        query_embedding = embedder.encode_text(query)
        search_results = search_engine.search(query_embedding, top_k=3)

        assert len(search_results) > 0

        # Step 2: Get recommendations for first search result
        selected_product_id = search_results[0]['product_id']
        recs = recommender.complementary_recommendations(
            selected_product_id,
            top_n=5
        )

        # Verify workflow
        assert isinstance(recs, list)

    def test_filter_and_search_workflow(self, setup_system):
        """Test filtering with search."""
        embedder = setup_system['embedder']
        search_engine = setup_system['search_engine']

        # Search with filters
        query = "dress"
        query_embedding = embedder.encode_text(query)

        results = search_engine.search_with_filters(
            query_embedding,
            top_k=5,
            filters={'gender': 'Women'}
        )

        # All results should match filter
        assert all(r['gender'] == 'Women' for r in results)


class TestDataPipeline:
    """Test data pipeline integration."""

    def test_loader_to_embeddings(self):
        """Test data loading to embedding generation."""
        # Load data
        loader = DatasetLoader()
        df = loader.create_sample_dataset(num_samples=20)

        # Generate embeddings
        embedder = CLIPEmbedder()
        descriptions = df.apply(
            lambda x: f"{x['category']} {x['baseColour']}",
            axis=1
        ).tolist()

        embeddings = embedder.encode_text(descriptions)

        # Verify
        assert len(embeddings) == len(df)
        assert embeddings.shape[1] == 512

    def test_full_pipeline_performance(self, setup_system):
        """Test performance of full pipeline."""
        import time

        embedder = setup_system['embedder']
        search_engine = setup_system['search_engine']

        # Time query processing
        query = "black shoes"

        start_time = time.time()

        # Generate embedding
        query_embedding = embedder.encode_text(query)

        # Search
        results = search_engine.search(query_embedding, top_k=10)

        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000

        # Should be fast (< 1 second for small dataset)
        assert elapsed_ms < 1000
        assert len(results) == 10
