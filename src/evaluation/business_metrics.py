"""
Business metrics for e-commerce evaluation.
Includes CTR, conversion rate, revenue metrics, and engagement metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class BusinessMetrics:
    """Calculate business and engagement metrics."""

    @staticmethod
    def click_through_rate(impressions: int, clicks: int) -> float:
        """
        Calculate Click-Through Rate (CTR).

        Args:
            impressions: Number of times shown
            clicks: Number of clicks

        Returns:
            CTR as percentage
        """
        if impressions == 0:
            return 0.0
        return (clicks / impressions) * 100

    @staticmethod
    def conversion_rate(clicks: int, conversions: int) -> float:
        """
        Calculate Conversion Rate.

        Args:
            clicks: Number of clicks
            conversions: Number of purchases/conversions

        Returns:
            Conversion rate as percentage
        """
        if clicks == 0:
            return 0.0
        return (conversions / clicks) * 100

    @staticmethod
    def add_to_cart_rate(views: int, add_to_cart: int) -> float:
        """
        Calculate Add-to-Cart Rate.

        Args:
            views: Number of product views
            add_to_cart: Number of add-to-cart actions

        Returns:
            Add-to-cart rate as percentage
        """
        if views == 0:
            return 0.0
        return (add_to_cart / views) * 100

    @staticmethod
    def bounce_rate(sessions: int, bounced_sessions: int) -> float:
        """
        Calculate Bounce Rate.

        Args:
            sessions: Total sessions
            bounced_sessions: Sessions with only one interaction

        Returns:
            Bounce rate as percentage
        """
        if sessions == 0:
            return 0.0
        return (bounced_sessions / sessions) * 100

    @staticmethod
    def average_order_value(total_revenue: float, num_orders: int) -> float:
        """
        Calculate Average Order Value (AOV).

        Args:
            total_revenue: Total revenue
            num_orders: Number of orders

        Returns:
            Average order value
        """
        if num_orders == 0:
            return 0.0
        return total_revenue / num_orders

    @staticmethod
    def customer_lifetime_value(
        avg_order_value: float,
        purchase_frequency: float,
        customer_lifespan_months: float
    ) -> float:
        """
        Calculate Customer Lifetime Value (CLV).

        Args:
            avg_order_value: Average order value
            purchase_frequency: Purchases per month
            customer_lifespan_months: Average customer lifespan in months

        Returns:
            Customer lifetime value
        """
        return avg_order_value * purchase_frequency * customer_lifespan_months

    @staticmethod
    def cart_abandonment_rate(
        carts_created: int,
        carts_purchased: int
    ) -> float:
        """
        Calculate Cart Abandonment Rate.

        Args:
            carts_created: Number of carts created
            carts_purchased: Number of carts that led to purchase

        Returns:
            Cart abandonment rate as percentage
        """
        if carts_created == 0:
            return 0.0
        return ((carts_created - carts_purchased) / carts_created) * 100

    @staticmethod
    def return_rate(items_sold: int, items_returned: int) -> float:
        """
        Calculate Return Rate.

        Args:
            items_sold: Number of items sold
            items_returned: Number of items returned

        Returns:
            Return rate as percentage
        """
        if items_sold == 0:
            return 0.0
        return (items_returned / items_sold) * 100

    @staticmethod
    def revenue_per_click(total_revenue: float, total_clicks: int) -> float:
        """
        Calculate Revenue Per Click (RPC).

        Args:
            total_revenue: Total revenue generated
            total_clicks: Total number of clicks

        Returns:
            Revenue per click
        """
        if total_clicks == 0:
            return 0.0
        return total_revenue / total_clicks

    @staticmethod
    def engagement_rate(
        interactions: int,
        impressions: int
    ) -> float:
        """
        Calculate Engagement Rate.

        Args:
            interactions: Number of interactions (clicks, likes, etc.)
            impressions: Number of impressions

        Returns:
            Engagement rate as percentage
        """
        if impressions == 0:
            return 0.0
        return (interactions / impressions) * 100


class FunnelAnalysis:
    """Analyze conversion funnels."""

    def __init__(self):
        self.funnel_stages = [
            "impression",
            "view",
            "click",
            "add_to_cart",
            "checkout",
            "purchase"
        ]

    def calculate_funnel_metrics(
        self,
        stage_counts: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Calculate funnel conversion rates between stages.

        Args:
            stage_counts: Dictionary mapping stage name to count

        Returns:
            Dictionary of conversion rates
        """
        metrics = {}

        for i in range(len(self.funnel_stages) - 1):
            current_stage = self.funnel_stages[i]
            next_stage = self.funnel_stages[i + 1]

            if current_stage in stage_counts and next_stage in stage_counts:
                current_count = stage_counts[current_stage]
                next_count = stage_counts[next_stage]

                if current_count > 0:
                    conversion = (next_count / current_count) * 100
                    metrics[f"{current_stage}_to_{next_stage}"] = conversion
                    metrics[f"{current_stage}_drop_off"] = 100 - conversion

        # Overall conversion from impression to purchase
        if "impression" in stage_counts and "purchase" in stage_counts:
            if stage_counts["impression"] > 0:
                metrics["overall_conversion"] = (
                    stage_counts["purchase"] / stage_counts["impression"]
                ) * 100

        return metrics

    def identify_bottlenecks(
        self,
        stage_counts: Dict[str, int]
    ) -> List[Tuple[str, float]]:
        """
        Identify funnel bottlenecks (stages with highest drop-off).

        Args:
            stage_counts: Dictionary mapping stage name to count

        Returns:
            List of (stage, drop_off_rate) tuples, sorted by drop-off
        """
        metrics = self.calculate_funnel_metrics(stage_counts)

        drop_offs = [
            (stage.replace("_drop_off", ""), rate)
            for stage, rate in metrics.items()
            if "_drop_off" in stage
        ]

        # Sort by drop-off rate (highest first)
        drop_offs.sort(key=lambda x: x[1], reverse=True)

        return drop_offs

    def plot_funnel(
        self,
        stage_counts: Dict[str, int],
        save_path: Optional[str] = None
    ):
        """
        Plot conversion funnel visualization.

        Args:
            stage_counts: Dictionary mapping stage name to count
            save_path: Optional path to save figure
        """
        stages = [s for s in self.funnel_stages if s in stage_counts]
        counts = [stage_counts[s] for s in stages]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create funnel visualization
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(stages)))

        for i, (stage, count) in enumerate(zip(stages, counts)):
            width = count / max(counts) * 0.8
            ax.barh(i, width, color=colors[i], edgecolor='black', linewidth=1.5)

            # Add count label
            ax.text(width + 0.02, i, f"{count:,}", va='center', fontsize=10)

            # Add stage label
            ax.text(-0.02, i, stage.replace('_', ' ').title(),
                   ha='right', va='center', fontsize=10, fontweight='bold')

        # Calculate and display conversion rates
        for i in range(len(stages) - 1):
            if counts[i] > 0:
                conversion = (counts[i + 1] / counts[i]) * 100
                ax.text(0.5, i + 0.5, f"↓ {conversion:.1f}%",
                       ha='center', va='center', fontsize=9, style='italic')

        ax.set_xlim(0, 1.2)
        ax.set_ylim(-0.5, len(stages) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.title('Conversion Funnel Analysis', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


class CohortAnalysis:
    """Analyze user cohorts over time."""

    def calculate_retention(
        self,
        cohort_data: pd.DataFrame,
        cohort_date_col: str,
        activity_date_col: str,
        user_id_col: str
    ) -> pd.DataFrame:
        """
        Calculate cohort retention rates.

        Args:
            cohort_data: DataFrame with user activity data
            cohort_date_col: Column name for cohort date (first activity)
            activity_date_col: Column name for activity date
            user_id_col: Column name for user ID

        Returns:
            DataFrame with retention rates
        """
        # Create cohort periods
        cohort_data['CohortPeriod'] = (
            (cohort_data[activity_date_col] - cohort_data[cohort_date_col])
            .dt.days // 30  # Monthly cohorts
        )

        # Group by cohort and period
        cohort_grouped = cohort_data.groupby(
            [cohort_date_col, 'CohortPeriod']
        )[user_id_col].nunique()

        cohort_counts = cohort_grouped.unstack(fill_value=0)

        # Calculate retention percentages
        cohort_sizes = cohort_counts.iloc[:, 0]
        retention = cohort_counts.divide(cohort_sizes, axis=0) * 100

        return retention

    def plot_cohort_heatmap(
        self,
        retention_data: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot cohort retention heatmap.

        Args:
            retention_data: DataFrame with retention rates
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(12, 8))

        sns.heatmap(
            retention_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=50,
            vmin=0,
            vmax=100,
            linewidths=0.5
        )

        plt.title('Cohort Retention Analysis (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Months Since First Activity', fontsize=12)
        plt.ylabel('Cohort', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


class RecommendationMetrics:
    """Metrics specific to recommendation systems."""

    @staticmethod
    def catalog_coverage(
        recommended_items: set,
        total_items: set
    ) -> float:
        """
        Calculate what percentage of catalog is being recommended.

        Args:
            recommended_items: Set of recommended item IDs
            total_items: Set of all item IDs in catalog

        Returns:
            Coverage as percentage
        """
        if len(total_items) == 0:
            return 0.0
        return (len(recommended_items) / len(total_items)) * 100

    @staticmethod
    def diversity_score(
        recommendations: List[Dict],
        attribute_key: str
    ) -> float:
        """
        Calculate diversity of recommendations.

        Args:
            recommendations: List of recommended items
            attribute_key: Attribute to measure diversity on

        Returns:
            Diversity score (0 to 1)
        """
        if not recommendations:
            return 0.0

        values = [r.get(attribute_key) for r in recommendations if attribute_key in r]
        if not values:
            return 0.0

        unique_values = len(set(values))
        return unique_values / len(values)

    @staticmethod
    def novelty_score(
        recommendations: List[str],
        popular_items: set
    ) -> float:
        """
        Calculate novelty (how many recommendations are non-popular items).

        Args:
            recommendations: List of recommended item IDs
            popular_items: Set of popular item IDs

        Returns:
            Novelty score (0 to 1)
        """
        if not recommendations:
            return 0.0

        novel_items = [r for r in recommendations if r not in popular_items]
        return len(novel_items) / len(recommendations)

    @staticmethod
    def serendipity_score(
        recommendations: List[str],
        expected_items: set,
        relevant_items: set
    ) -> float:
        """
        Calculate serendipity (unexpected but relevant recommendations).

        Args:
            recommendations: List of recommended item IDs
            expected_items: Set of expected/obvious items
            relevant_items: Set of relevant items

        Returns:
            Serendipity score (0 to 1)
        """
        if not recommendations:
            return 0.0

        serendipitous = [
            r for r in recommendations
            if r in relevant_items and r not in expected_items
        ]

        return len(serendipitous) / len(recommendations)


def calculate_ab_test_significance(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate statistical significance for A/B test.

    Args:
        control_conversions: Conversions in control group
        control_total: Total in control group
        treatment_conversions: Conversions in treatment group
        treatment_total: Total in treatment group
        confidence_level: Desired confidence level

    Returns:
        Dictionary with test results
    """
    from scipy import stats

    # Calculate proportions
    p1 = control_conversions / control_total if control_total > 0 else 0
    p2 = treatment_conversions / treatment_total if treatment_total > 0 else 0

    # Pooled proportion
    p_pool = (control_conversions + treatment_conversions) / (control_total + treatment_total)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))

    # Z-score
    z = (p2 - p1) / se if se > 0 else 0

    # P-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Confidence interval
    ci_z = stats.norm.ppf((1 + confidence_level) / 2)
    ci_diff = ci_z * se
    ci_lower = (p2 - p1) - ci_diff
    ci_upper = (p2 - p1) + ci_diff

    # Lift
    lift = ((p2 - p1) / p1 * 100) if p1 > 0 else 0

    return {
        "control_rate": p1 * 100,
        "treatment_rate": p2 * 100,
        "lift_percentage": lift,
        "p_value": p_value,
        "is_significant": p_value < (1 - confidence_level),
        "confidence_interval_lower": ci_lower * 100,
        "confidence_interval_upper": ci_upper * 100,
        "z_score": z
    }
