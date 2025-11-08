"""
Visual search engine using FAISS for efficient similarity search.
"""

import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
import pandas as pd


class VisualSearchEngine:
    """Efficient visual search using FAISS index."""

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize search engine.

        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.product_ids = []
        self.metadata = None

    def build_index(
        self,
        embeddings: np.ndarray,
        product_ids: List[str],
        metadata: Optional[pd.DataFrame] = None,
        index_type: str = "l2"
    ):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Array of embeddings (N x embedding_dim)
            product_ids: List of product IDs corresponding to embeddings
            metadata: Optional DataFrame with product metadata
            index_type: 'l2' or 'cosine' similarity
        """
        assert len(embeddings) == len(product_ids), "Embeddings and IDs must match"

        self.product_ids = product_ids
        self.metadata = metadata

        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')

        # Normalize for cosine similarity if needed
        if index_type == "cosine":
            faiss.normalize_L2(embeddings)

        # Build index
        if len(embeddings) < 1000:
            # Use flat index for small datasets
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            # Use IVF index for larger datasets
            nlist = min(100, len(embeddings) // 10)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)

        # Add vectors to index
        self.index.add(embeddings)

        print(f"✓ Built FAISS index with {len(embeddings)} vectors")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar products.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Optional similarity threshold

        Returns:
            List of results with product info and scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Ensure query is 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue

            # Convert distance to similarity score (0 to 1)
            # For L2 distance, similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + float(dist))

            if threshold is not None and similarity < threshold:
                continue

            result = {
                'product_id': self.product_ids[idx],
                'similarity': similarity,
                'distance': float(dist)
            }

            # Add metadata if available
            if self.metadata is not None:
                product_data = self.metadata[
                    self.metadata['id'] == self.product_ids[idx]
                ]
                if len(product_data) > 0:
                    result.update(product_data.iloc[0].to_dict())

            results.append(result)

        return results

    def search_with_filters(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search with attribute filters.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Dictionary of attribute filters (e.g., {'category': 'Dress'})

        Returns:
            List of filtered results
        """
        # Get more candidates than needed
        candidates = self.search(query_embedding, top_k=top_k * 5)

        if filters is None or self.metadata is None:
            return candidates[:top_k]

        # Apply filters
        filtered_results = []
        for result in candidates:
            match = True
            for key, value in filters.items():
                if key in result and result[key] != value:
                    match = False
                    break

            if match:
                filtered_results.append(result)

            if len(filtered_results) >= top_k:
                break

        return filtered_results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[List[Dict]]:
        """
        Batch search for multiple queries.

        Args:
            query_embeddings: Array of query embeddings (N x embedding_dim)
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        query_embeddings = query_embeddings.astype('float32')

        distances, indices = self.index.search(query_embeddings, top_k)

        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_distances, query_indices):
                if idx == -1:
                    continue

                similarity = 1.0 / (1.0 + float(dist))

                result = {
                    'product_id': self.product_ids[idx],
                    'similarity': similarity,
                    'distance': float(dist)
                }

                if self.metadata is not None:
                    product_data = self.metadata[
                        self.metadata['id'] == self.product_ids[idx]
                    ]
                    if len(product_data) > 0:
                        result.update(product_data.iloc[0].to_dict())

                results.append(result)

            all_results.append(results)

        return all_results

    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, filepath)

        # Save metadata
        metadata_path = filepath + '.metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'product_ids': self.product_ids,
                'metadata': self.metadata
            }, f)

        print(f"✓ Saved index to {filepath}")

    def load_index(self, filepath: str):
        """Load FAISS index and metadata from disk."""
        # Load index
        self.index = faiss.read_index(filepath)

        # Load metadata
        metadata_path = filepath + '.metadata.pkl'
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.product_ids = data['product_ids']
            self.metadata = data['metadata']

        print(f"✓ Loaded index from {filepath}")

    def get_statistics(self) -> Dict:
        """Get search engine statistics."""
        if self.index is None:
            return {'status': 'Index not built'}

        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'total_products': len(self.product_ids)
        }
