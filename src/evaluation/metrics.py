"""
Evaluation metrics for search and recommendation systems.
"""

import numpy as np
from typing import List, Dict, Set
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SearchMetrics:
    """Metrics for evaluating search quality."""

    @staticmethod
    def precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        """
        Calculate Precision@K.

        Args:
            relevant: Set of relevant product IDs
            retrieved: List of retrieved product IDs (ordered by rank)
            k: Cutoff position

        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & relevant)

        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        """
        Calculate Recall@K.

        Args:
            relevant: Set of relevant product IDs
            retrieved: List of retrieved product IDs
            k: Cutoff position

        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & relevant)

        return relevant_retrieved / len(relevant)

    @staticmethod
    def average_precision(relevant: Set[str], retrieved: List[str]) -> float:
        """
        Calculate Average Precision (AP).

        Args:
            relevant: Set of relevant product IDs
            retrieved: List of retrieved product IDs

        Returns:
            Average Precision score
        """
        if len(relevant) == 0:
            return 0.0

        num_hits = 0.0
        sum_precisions = 0.0

        for i, item in enumerate(retrieved):
            if item in relevant:
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)
                sum_precisions += precision_at_i

        if num_hits == 0:
            return 0.0

        return sum_precisions / len(relevant)

    @staticmethod
    def mean_average_precision(
        relevance_dict: Dict[str, Set[str]],
        results_dict: Dict[str, List[str]]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP).

        Args:
            relevance_dict: Dict mapping query_id to set of relevant items
            results_dict: Dict mapping query_id to list of retrieved items

        Returns:
            MAP score
        """
        aps = []

        for query_id in relevance_dict:
            if query_id in results_dict:
                relevant = relevance_dict[query_id]
                retrieved = results_dict[query_id]
                ap = SearchMetrics.average_precision(relevant, retrieved)
                aps.append(ap)

        return np.mean(aps) if aps else 0.0

    @staticmethod
    def reciprocal_rank(relevant: Set[str], retrieved: List[str]) -> float:
        """
        Calculate Reciprocal Rank (RR).

        Args:
            relevant: Set of relevant product IDs
            retrieved: List of retrieved product IDs

        Returns:
            Reciprocal Rank score
        """
        for i, item in enumerate(retrieved):
            if item in relevant:
                return 1.0 / (i + 1.0)

        return 0.0

    @staticmethod
    def mean_reciprocal_rank(
        relevance_dict: Dict[str, Set[str]],
        results_dict: Dict[str, List[str]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            relevance_dict: Dict mapping query_id to set of relevant items
            results_dict: Dict mapping query_id to list of retrieved items

        Returns:
            MRR score
        """
        rrs = []

        for query_id in relevance_dict:
            if query_id in results_dict:
                relevant = relevance_dict[query_id]
                retrieved = results_dict[query_id]
                rr = SearchMetrics.reciprocal_rank(relevant, retrieved)
                rrs.append(rr)

        return np.mean(rrs) if rrs else 0.0

    @staticmethod
    def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Args:
            relevant: Set of relevant product IDs
            retrieved: List of retrieved product IDs
            k: Cutoff position

        Returns:
            NDCG@K score
        """
        retrieved_at_k = retrieved[:k]

        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(retrieved_at_k):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # +2 because index starts at 0

        # Calculate IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

        if idcg == 0:
            return 0.0

        return dcg / idcg


class RecommendationMetrics:
    """Metrics for evaluating recommendation quality."""

    @staticmethod
    def diversity_score(recommendations: List[Dict], attribute: str = 'category') -> float:
        """
        Calculate diversity of recommendations based on an attribute.

        Args:
            recommendations: List of recommended products
            attribute: Attribute to measure diversity on

        Returns:
            Diversity score (0 to 1)
        """
        if len(recommendations) == 0:
            return 0.0

        # Get unique values for the attribute
        values = [rec.get(attribute) for rec in recommendations if attribute in rec]

        if len(values) == 0:
            return 0.0

        unique_values = len(set(values))

        return unique_values / len(values)

    @staticmethod
    def coverage(
        all_products: Set[str],
        recommended_products: Set[str]
    ) -> float:
        """
        Calculate catalog coverage.

        Args:
            all_products: Set of all product IDs
            recommended_products: Set of products that were recommended

        Returns:
            Coverage score (0 to 1)
        """
        if len(all_products) == 0:
            return 0.0

        return len(recommended_products) / len(all_products)

    @staticmethod
    def novelty_score(recommendations: List[Dict], popularity: Dict[str, float]) -> float:
        """
        Calculate novelty (how unexpected/non-obvious recommendations are).

        Args:
            recommendations: List of recommended products
            popularity: Dict mapping product_id to popularity score

        Returns:
            Average novelty score
        """
        if len(recommendations) == 0:
            return 0.0

        novelties = []
        for rec in recommendations:
            product_id = rec.get('id')
            if product_id in popularity:
                # Novelty is inverse of popularity
                novelties.append(1.0 - popularity[product_id])

        return np.mean(novelties) if novelties else 0.0


class VisualizationUtils:
    """Utilities for visualizing evaluation results."""

    @staticmethod
    def plot_precision_recall_curve(
        precisions: List[float],
        recalls: List[float],
        save_path: Optional[str] = None
    ):
        """Plot Precision-Recall curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, marker='o', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_metrics_at_k(
        k_values: List[int],
        metrics_dict: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """
        Plot metrics at different K values.

        Args:
            k_values: List of K values
            metrics_dict: Dict mapping metric name to list of values
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(12, 6))

        for metric_name, values in metrics_dict.items():
            plt.plot(k_values, values, marker='o', linewidth=2, label=metric_name)

        plt.xlabel('K', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Metrics at Different K Values', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str],
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_similarity_distribution(
        similarities: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot distribution of similarity scores."""
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Similarity Scores', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


from typing import Optional

def evaluate_search_system(
    relevance_dict: Dict[str, Set[str]],
    results_dict: Dict[str, List[str]],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Comprehensive evaluation of search system.

    Args:
        relevance_dict: Ground truth relevance
        results_dict: Search results
        k_values: K values to evaluate at

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Calculate MAP and MRR
    metrics['MAP'] = SearchMetrics.mean_average_precision(relevance_dict, results_dict)
    metrics['MRR'] = SearchMetrics.mean_reciprocal_rank(relevance_dict, results_dict)

    # Calculate metrics at different K values
    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []

        for query_id in relevance_dict:
            if query_id in results_dict:
                relevant = relevance_dict[query_id]
                retrieved = results_dict[query_id]

                precisions.append(SearchMetrics.precision_at_k(relevant, retrieved, k))
                recalls.append(SearchMetrics.recall_at_k(relevant, retrieved, k))
                ndcgs.append(SearchMetrics.ndcg_at_k(relevant, retrieved, k))

        metrics[f'Precision@{k}'] = np.mean(precisions) if precisions else 0.0
        metrics[f'Recall@{k}'] = np.mean(recalls) if recalls else 0.0
        metrics[f'NDCG@{k}'] = np.mean(ndcgs) if ndcgs else 0.0

    return metrics
