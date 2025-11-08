"""
Unit tests for business metrics.
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.business_metrics import (
    BusinessMetrics, FunnelAnalysis, RecommendationMetrics
)


class TestBusinessMetrics:
    """Test business metrics calculations."""

    def test_click_through_rate(self):
        """Test CTR calculation."""
        ctr = BusinessMetrics.click_through_rate(
            impressions=1000,
            clicks=50
        )
        assert ctr == 5.0

        # Zero impressions
        ctr_zero = BusinessMetrics.click_through_rate(
            impressions=0,
            clicks=0
        )
        assert ctr_zero == 0.0

    def test_conversion_rate(self):
        """Test conversion rate calculation."""
        cr = BusinessMetrics.conversion_rate(
            clicks=100,
            conversions=10
        )
        assert cr == 10.0

        # Zero clicks
        cr_zero = BusinessMetrics.conversion_rate(
            clicks=0,
            conversions=0
        )
        assert cr_zero == 0.0

    def test_add_to_cart_rate(self):
        """Test add-to-cart rate calculation."""
        atc = BusinessMetrics.add_to_cart_rate(
            views=200,
            add_to_cart=40
        )
        assert atc == 20.0

    def test_average_order_value(self):
        """Test AOV calculation."""
        aov = BusinessMetrics.average_order_value(
            total_revenue=5000.0,
            num_orders=100
        )
        assert aov == 50.0

    def test_cart_abandonment_rate(self):
        """Test cart abandonment rate."""
        car = BusinessMetrics.cart_abandonment_rate(
            carts_created=100,
            carts_purchased=75
        )
        assert car == 25.0

    def test_revenue_per_click(self):
        """Test RPC calculation."""
        rpc = BusinessMetrics.revenue_per_click(
            total_revenue=10000.0,
            total_clicks=500
        )
        assert rpc == 20.0


class TestFunnelAnalysis:
    """Test funnel analysis functionality."""

    @pytest.fixture
    def funnel(self):
        """Create funnel analyzer."""
        return FunnelAnalysis()

    @pytest.fixture
    def sample_funnel_data(self):
        """Create sample funnel data."""
        return {
            "impression": 10000,
            "view": 5000,
            "click": 1000,
            "add_to_cart": 300,
            "checkout": 200,
            "purchase": 150
        }

    def test_calculate_funnel_metrics(self, funnel, sample_funnel_data):
        """Test funnel metrics calculation."""
        metrics = funnel.calculate_funnel_metrics(sample_funnel_data)

        assert "impression_to_view" in metrics
        assert "overall_conversion" in metrics

        # Check specific conversion
        assert metrics["impression_to_view"] == 50.0
        assert metrics["overall_conversion"] == 1.5

    def test_identify_bottlenecks(self, funnel, sample_funnel_data):
        """Test bottleneck identification."""
        bottlenecks = funnel.identify_bottlenecks(sample_funnel_data)

        assert len(bottlenecks) > 0
        # Bottlenecks should be sorted by drop-off rate
        rates = [rate for _, rate in bottlenecks]
        assert rates == sorted(rates, reverse=True)


class TestRecommendationMetrics:
    """Test recommendation-specific metrics."""

    def test_catalog_coverage(self):
        """Test catalog coverage calculation."""
        recommended = set(["A", "B", "C"])
        total = set(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])

        coverage = RecommendationMetrics.catalog_coverage(
            recommended, total
        )
        assert coverage == 30.0

    def test_diversity_score(self):
        """Test diversity score calculation."""
        recommendations = [
            {"category": "Dress"},
            {"category": "Shirt"},
            {"category": "Dress"},
            {"category": "Jeans"}
        ]

        diversity = RecommendationMetrics.diversity_score(
            recommendations,
            "category"
        )
        assert diversity == 0.75  # 3 unique / 4 total

    def test_novelty_score(self):
        """Test novelty score calculation."""
        recommendations = ["A", "B", "C", "D", "E"]
        popular_items = set(["A", "B"])

        novelty = RecommendationMetrics.novelty_score(
            recommendations,
            popular_items
        )
        assert novelty == 0.6  # 3 novel / 5 total
