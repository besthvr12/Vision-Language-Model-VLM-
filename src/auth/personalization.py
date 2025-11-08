"""
Personalization engine for tailored user experiences.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from collections import Counter
from datetime import datetime, timedelta

from .user_manager import UserPreferences, UserInteraction


class PersonalizationEngine:
    """Provides personalized recommendations and experiences."""

    def __init__(self):
        pass

    def get_personalized_weights(
        self,
        user_preferences: UserPreferences,
        user_history: List[UserInteraction]
    ) -> Dict[str, float]:
        """
        Calculate personalization weights based on user preferences and history.

        Returns:
            Dictionary of attribute weights for recommendation scoring
        """
        weights = {
            "category": 1.0,
            "color": 1.0,
            "gender": 1.0,
            "price": 1.0,
            "recency": 1.0
        }

        # Boost weights based on preferences
        if user_preferences.favorite_categories:
            weights["category"] = 2.0

        if user_preferences.favorite_colors:
            weights["color"] = 1.5

        if user_preferences.preferred_gender:
            weights["gender"] = 1.5

        # Adjust based on interaction history
        if user_history:
            # Recent activity boosts recency weight
            recent_interactions = [
                i for i in user_history
                if i.timestamp > datetime.utcnow() - timedelta(days=7)
            ]
            if len(recent_interactions) > 10:
                weights["recency"] = 2.0

        return weights

    def score_product_for_user(
        self,
        product: Dict,
        user_preferences: UserPreferences,
        user_history: List[UserInteraction],
        base_score: float = 0.0
    ) -> float:
        """
        Score a product for personalized ranking.

        Args:
            product: Product dictionary
            user_preferences: User preferences
            user_history: User interaction history
            base_score: Base relevance score (e.g., from search)

        Returns:
            Personalized score
        """
        score = base_score

        # Get personalization weights
        weights = self.get_personalized_weights(user_preferences, user_history)

        # Category match
        if product.get("category") in user_preferences.favorite_categories:
            score += 0.3 * weights["category"]

        # Color match
        if product.get("baseColour") in user_preferences.favorite_colors:
            score += 0.2 * weights["color"]

        # Gender match
        if product.get("gender") == user_preferences.preferred_gender:
            score += 0.15 * weights["gender"]

        # Price range match
        price = product.get("price", 0)
        if user_preferences.price_range_min and user_preferences.price_range_max:
            if user_preferences.price_range_min <= price <= user_preferences.price_range_max:
                score += 0.15 * weights["price"]

        # Brand match
        if product.get("brand") in user_preferences.preferred_brands:
            score += 0.1

        # Interaction history boost
        product_id = product.get("id")
        viewed_products = [i.product_id for i in user_history if i.interaction_type == "view"]
        if product_id in viewed_products:
            # User has viewed this before but didn't purchase
            score += 0.05 * weights["recency"]

        return score

    def get_trending_for_user(
        self,
        all_interactions: List[UserInteraction],
        user_preferences: UserPreferences,
        time_window_hours: int = 24,
        top_n: int = 10
    ) -> List[str]:
        """
        Get trending products personalized for user.

        Args:
            all_interactions: All user interactions in the system
            user_preferences: User's preferences
            time_window_hours: Time window for trending calculation
            top_n: Number of trending items to return

        Returns:
            List of trending product IDs
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        # Get recent interactions
        recent = [
            i for i in all_interactions
            if i.timestamp > cutoff_time
        ]

        # Count product interactions
        product_counts = Counter([i.product_id for i in recent])

        # Get top trending
        trending = [pid for pid, _ in product_counts.most_common(top_n * 2)]

        return trending[:top_n]

    def get_collaborative_recommendations(
        self,
        user_id: str,
        all_interactions: List[UserInteraction],
        top_n: int = 10
    ) -> List[str]:
        """
        Collaborative filtering recommendations.

        Args:
            user_id: Target user ID
            all_interactions: All interactions in system
            top_n: Number of recommendations

        Returns:
            List of recommended product IDs
        """
        # Get user's purchased products
        user_purchases = [
            i.product_id for i in all_interactions
            if i.user_id == user_id and i.interaction_type == "purchase"
        ]

        if not user_purchases:
            return []

        # Find similar users (users who bought the same products)
        similar_users = set()
        for interaction in all_interactions:
            if (interaction.interaction_type == "purchase" and
                interaction.product_id in user_purchases and
                interaction.user_id != user_id):
                similar_users.add(interaction.user_id)

        # Get products purchased by similar users
        recommendations = []
        for interaction in all_interactions:
            if (interaction.user_id in similar_users and
                interaction.interaction_type == "purchase" and
                interaction.product_id not in user_purchases):
                recommendations.append(interaction.product_id)

        # Count and rank
        product_counts = Counter(recommendations)
        top_products = [pid for pid, _ in product_counts.most_common(top_n)]

        return top_products

    def rerank_results(
        self,
        results: List[Dict],
        user_preferences: UserPreferences,
        user_history: List[UserInteraction]
    ) -> List[Dict]:
        """
        Re-rank search/recommendation results based on personalization.

        Args:
            results: Original results with scores
            user_preferences: User preferences
            user_history: User history

        Returns:
            Re-ranked results
        """
        personalized_results = []

        for result in results:
            base_score = result.get("similarity", result.get("score", 0.5))

            personalized_score = self.score_product_for_user(
                result,
                user_preferences,
                user_history,
                base_score
            )

            result_copy = result.copy()
            result_copy["personalized_score"] = personalized_score
            result_copy["original_score"] = base_score
            personalized_results.append(result_copy)

        # Sort by personalized score
        personalized_results.sort(key=lambda x: x["personalized_score"], reverse=True)

        return personalized_results

    def get_user_insights(
        self,
        user_history: List[UserInteraction]
    ) -> Dict:
        """
        Extract insights from user behavior.

        Args:
            user_history: User's interaction history

        Returns:
            Dictionary of insights
        """
        if not user_history:
            return {}

        insights = {}

        # Most viewed categories
        viewed_products = [
            i.metadata.get("category") for i in user_history
            if i.interaction_type == "view" and "category" in i.metadata
        ]
        if viewed_products:
            insights["top_categories"] = [
                cat for cat, _ in Counter(viewed_products).most_common(5)
            ]

        # Most viewed colors
        viewed_colors = [
            i.metadata.get("color") for i in user_history
            if i.interaction_type == "view" and "color" in i.metadata
        ]
        if viewed_colors:
            insights["top_colors"] = [
                color for color, _ in Counter(viewed_colors).most_common(5)
            ]

        # Average price of viewed items
        viewed_prices = [
            i.metadata.get("price") for i in user_history
            if i.interaction_type == "view" and "price" in i.metadata
        ]
        if viewed_prices:
            insights["avg_viewed_price"] = np.mean(viewed_prices)
            insights["price_range"] = {
                "min": np.min(viewed_prices),
                "max": np.max(viewed_prices)
            }

        # Browsing patterns
        insights["browsing_pattern"] = {
            "total_sessions": len(set([i.session_id for i in user_history if i.session_id])),
            "avg_items_per_session": len(user_history) / max(len(set([i.session_id for i in user_history if i.session_id])), 1),
            "conversion_funnel": {
                "views": len([i for i in user_history if i.interaction_type == "view"]),
                "clicks": len([i for i in user_history if i.interaction_type == "click"]),
                "add_to_cart": len([i for i in user_history if i.interaction_type == "add_to_cart"]),
                "purchases": len([i for i in user_history if i.interaction_type == "purchase"])
            }
        }

        # Time-based patterns
        hours = [i.timestamp.hour for i in user_history]
        if hours:
            insights["peak_hours"] = [
                hour for hour, _ in Counter(hours).most_common(3)
            ]

        return insights

    def auto_update_preferences(
        self,
        current_preferences: UserPreferences,
        user_history: List[UserInteraction]
    ) -> UserPreferences:
        """
        Automatically update user preferences based on behavior.

        Args:
            current_preferences: Current user preferences
            user_history: User interaction history

        Returns:
            Updated preferences
        """
        insights = self.get_user_insights(user_history)

        # Update favorite categories
        if "top_categories" in insights:
            current_preferences.favorite_categories = insights["top_categories"][:5]

        # Update favorite colors
        if "top_colors" in insights:
            current_preferences.favorite_colors = insights["top_colors"][:5]

        # Update price range
        if "price_range" in insights:
            current_preferences.price_range_min = insights["price_range"]["min"]
            current_preferences.price_range_max = insights["price_range"]["max"]

        return current_preferences
