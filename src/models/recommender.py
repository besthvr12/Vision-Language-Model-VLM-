"""
Recommendation engine combining multiple strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationEngine:
    """Hybrid recommendation system for products."""

    def __init__(self, metadata: pd.DataFrame):
        """
        Initialize recommendation engine.

        Args:
            metadata: Product metadata DataFrame
        """
        self.metadata = metadata

    def content_based_recommendations(
        self,
        product_id: str,
        embeddings: np.ndarray,
        product_ids: List[str],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Content-based recommendations using embeddings.

        Args:
            product_id: Source product ID
            embeddings: All product embeddings
            product_ids: List of all product IDs
            top_n: Number of recommendations

        Returns:
            List of recommended products
        """
        # Find product index
        try:
            idx = product_ids.index(product_id)
        except ValueError:
            return []

        # Get product embedding
        product_emb = embeddings[idx].reshape(1, -1)

        # Compute similarities
        similarities = cosine_similarity(product_emb, embeddings)[0]

        # Get top similar (excluding itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

        # Build recommendations
        recommendations = []
        for sim_idx in similar_indices:
            rec_product_id = product_ids[sim_idx]
            product_data = self.metadata[self.metadata['id'] == rec_product_id]

            if len(product_data) > 0:
                rec = product_data.iloc[0].to_dict()
                rec['similarity'] = float(similarities[sim_idx])
                rec['reason'] = 'Similar style and features'
                recommendations.append(rec)

        return recommendations

    def attribute_based_recommendations(
        self,
        product_id: str,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Recommendations based on matching attributes.

        Args:
            product_id: Source product ID
            top_n: Number of recommendations

        Returns:
            List of recommended products
        """
        # Get source product
        source = self.metadata[self.metadata['id'] == product_id]
        if len(source) == 0:
            return []

        source = source.iloc[0]

        # Find products with matching attributes
        candidates = self.metadata[self.metadata['id'] != product_id].copy()

        # Score based on attribute matches
        candidates['score'] = 0

        # Same category = +3
        if 'category' in source:
            candidates.loc[candidates['category'] == source['category'], 'score'] += 3

        # Same color = +2
        if 'baseColour' in source:
            candidates.loc[candidates['baseColour'] == source['baseColour'], 'score'] += 2

        # Same gender = +2
        if 'gender' in source:
            candidates.loc[candidates['gender'] == source['gender'], 'score'] += 2

        # Same season = +1
        if 'season' in source:
            candidates.loc[candidates['season'] == source['season'], 'score'] += 1

        # Similar price range = +1
        if 'price' in source:
            price_diff = abs(candidates['price'] - source['price'])
            candidates.loc[price_diff < source['price'] * 0.2, 'score'] += 1

        # Sort by score
        candidates = candidates.sort_values('score', ascending=False)

        # Get top recommendations
        recommendations = []
        for _, row in candidates.head(top_n).iterrows():
            rec = row.to_dict()

            # Add reason
            reasons = []
            if 'category' in source and row['category'] == source['category']:
                reasons.append(f"Same category: {row['category']}")
            if 'baseColour' in source and row['baseColour'] == source['baseColour']:
                reasons.append(f"Matching color: {row['baseColour']}")
            if 'gender' in source and row['gender'] == source['gender']:
                reasons.append(f"For {row['gender']}")

            rec['reason'] = ', '.join(reasons) if reasons else 'Similar attributes'
            rec['score'] = row['score']

            recommendations.append(rec)

        return recommendations

    def complementary_recommendations(
        self,
        product_id: str,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Recommend complementary products (complete the look).

        Args:
            product_id: Source product ID
            top_n: Number of recommendations

        Returns:
            List of complementary products
        """
        # Get source product
        source = self.metadata[self.metadata['id'] == product_id]
        if len(source) == 0:
            return []

        source = source.iloc[0]

        # Define complementary categories
        complementary_map = {
            'Dress': ['Shoes', 'Accessories', 'Jacket'],
            'Shirt': ['Jeans', 'Shoes', 'Accessories'],
            'Jeans': ['Shirt', 'Shoes', 'Jacket'],
            'Shoes': ['Jeans', 'Shirt', 'Dress'],
            'Jacket': ['Jeans', 'Shirt', 'Dress'],
            'Accessories': ['Dress', 'Shirt', 'Jeans']
        }

        if 'category' not in source:
            return []

        # Get complementary categories
        comp_categories = complementary_map.get(source['category'], [])
        if not comp_categories:
            return []

        # Find products in complementary categories
        candidates = self.metadata[
            (self.metadata['category'].isin(comp_categories)) &
            (self.metadata['id'] != product_id)
        ].copy()

        # Score based on compatibility
        candidates['score'] = 0

        # Same gender = +3
        if 'gender' in source:
            candidates.loc[candidates['gender'] == source['gender'], 'score'] += 3

        # Same color scheme (color coordination) = +2
        if 'baseColour' in source:
            # Complementary colors get bonus
            color_complements = {
                'Red': ['Black', 'White', 'Blue'],
                'Blue': ['White', 'Black', 'Red'],
                'Black': ['Red', 'Blue', 'White', 'Yellow'],
                'White': ['Black', 'Blue', 'Red'],
                'Green': ['Brown', 'Black', 'White'],
                'Yellow': ['Black', 'Blue', 'White']
            }
            comp_colors = color_complements.get(source['baseColour'], [])
            candidates.loc[candidates['baseColour'].isin(comp_colors), 'score'] += 2

        # Same season = +1
        if 'season' in source:
            candidates.loc[candidates['season'] == source['season'], 'score'] += 1

        # Sort by score
        candidates = candidates.sort_values('score', ascending=False)

        # Get top recommendations
        recommendations = []
        for _, row in candidates.head(top_n).iterrows():
            rec = row.to_dict()
            rec['reason'] = f"Complements your {source.get('category', 'item')}"
            rec['score'] = row['score']
            recommendations.append(rec)

        return recommendations

    def hybrid_recommendations(
        self,
        product_id: str,
        embeddings: np.ndarray,
        product_ids: List[str],
        top_n: int = 10,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Hybrid recommendations combining multiple strategies.

        Args:
            product_id: Source product ID
            embeddings: All product embeddings
            product_ids: List of all product IDs
            top_n: Number of recommendations
            weights: Strategy weights (default: equal)

        Returns:
            List of recommended products
        """
        if weights is None:
            weights = {
                'content': 0.4,
                'attribute': 0.3,
                'complementary': 0.3
            }

        # Get recommendations from each strategy
        content_recs = self.content_based_recommendations(
            product_id, embeddings, product_ids, top_n=20
        )
        attribute_recs = self.attribute_based_recommendations(
            product_id, top_n=20
        )
        complementary_recs = self.complementary_recommendations(
            product_id, top_n=20
        )

        # Combine scores
        all_products = {}

        # Add content-based
        for i, rec in enumerate(content_recs):
            pid = rec['id']
            score = weights['content'] * (1.0 - i / len(content_recs))
            all_products[pid] = {
                'data': rec,
                'score': score,
                'sources': ['content']
            }

        # Add attribute-based
        for i, rec in enumerate(attribute_recs):
            pid = rec['id']
            score = weights['attribute'] * (1.0 - i / len(attribute_recs))

            if pid in all_products:
                all_products[pid]['score'] += score
                all_products[pid]['sources'].append('attribute')
            else:
                all_products[pid] = {
                    'data': rec,
                    'score': score,
                    'sources': ['attribute']
                }

        # Add complementary
        for i, rec in enumerate(complementary_recs):
            pid = rec['id']
            score = weights['complementary'] * (1.0 - i / len(complementary_recs))

            if pid in all_products:
                all_products[pid]['score'] += score
                all_products[pid]['sources'].append('complementary')
            else:
                all_products[pid] = {
                    'data': rec,
                    'score': score,
                    'sources': ['complementary']
                }

        # Sort by combined score
        sorted_products = sorted(
            all_products.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # Format results
        recommendations = []
        for pid, info in sorted_products[:top_n]:
            rec = info['data'].copy()
            rec['recommendation_score'] = info['score']
            rec['recommendation_sources'] = ', '.join(info['sources'])
            recommendations.append(rec)

        return recommendations

    def trending_products(self, top_n: int = 10, category: Optional[str] = None) -> List[Dict]:
        """
        Get trending products (placeholder - would use real engagement data).

        Args:
            top_n: Number of products to return
            category: Optional category filter

        Returns:
            List of trending products
        """
        df = self.metadata.copy()

        if category:
            df = df[df['category'] == category]

        # Simulate trending score (would use real metrics in production)
        df['trending_score'] = np.random.random(len(df))

        trending = df.sort_values('trending_score', ascending=False).head(top_n)

        results = []
        for _, row in trending.iterrows():
            result = row.to_dict()
            result['reason'] = 'Trending now'
            results.append(result)

        return results
