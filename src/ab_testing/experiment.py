"""
A/B Testing Framework for experimentation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
import random
import hashlib
from scipy import stats
import numpy as np


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class VariantType(str, Enum):
    CONTROL = "control"
    TREATMENT = "treatment"


class Variant(BaseModel):
    """A variant in an A/B test."""
    variant_id: str
    name: str
    variant_type: VariantType
    traffic_allocation: float  # Percentage of traffic (0.0 to 1.0)
    config: Dict[str, Any] = {}  # Configuration for this variant


class ExperimentMetric(BaseModel):
    """Metric tracked in an experiment."""
    metric_name: str
    metric_type: str  # count, rate, revenue, time
    primary: bool = False  # Is this the primary metric?


class Experiment(BaseModel):
    """A/B test experiment configuration."""
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: List[Variant]
    metrics: List[ExperimentMetric]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_at: datetime = datetime.utcnow()
    created_by: str
    targeting_rules: Dict[str, Any] = {}  # User targeting criteria


class ExperimentAssignment(BaseModel):
    """Record of user assignment to variant."""
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime = datetime.utcnow()
    session_id: Optional[str] = None


class ExperimentEvent(BaseModel):
    """Event tracked for experiment analysis."""
    user_id: str
    experiment_id: str
    variant_id: str
    metric_name: str
    value: float
    timestamp: datetime = datetime.utcnow()
    metadata: Dict[str, Any] = {}


class ExperimentResult(BaseModel):
    """Results of an A/B test."""
    experiment_id: str
    variant_results: Dict[str, Dict[str, float]]  # variant_id -> metrics
    statistical_significance: Dict[str, bool]  # metric -> is significant
    confidence_level: float = 0.95
    sample_sizes: Dict[str, int]  # variant_id -> sample size
    recommendation: str
    computed_at: datetime = datetime.utcnow()


class ABTestingFramework:
    """Manages A/B testing experiments."""

    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.assignments: List[ExperimentAssignment] = []
        self.events: List[ExperimentEvent] = []

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Variant],
        metrics: List[ExperimentMetric],
        created_by: str,
        targeting_rules: Optional[Dict] = None
    ) -> Experiment:
        """Create a new A/B test experiment."""
        experiment_id = f"exp_{len(self.experiments) + 1}"

        # Validate traffic allocation
        total_allocation = sum(v.traffic_allocation for v in variants)
        if not (0.99 <= total_allocation <= 1.01):  # Allow small floating point error
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            metrics=metrics,
            created_by=created_by,
            targeting_rules=targeting_rules or {}
        )

        self.experiments[experiment_id] = experiment
        return experiment

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.utcnow()
        return experiment

    def stop_experiment(self, experiment_id: str) -> Experiment:
        """Stop an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.utcnow()
        return experiment

    def _hash_user_to_variant(
        self,
        user_id: str,
        experiment_id: str,
        variants: List[Variant]
    ) -> str:
        """
        Consistently hash user to a variant.
        Same user always gets same variant for same experiment.
        """
        # Create deterministic hash
        hash_input = f"{user_id}:{experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Map to variant based on traffic allocation
        random_value = (hash_value % 10000) / 10000.0  # 0.0 to 1.0

        cumulative = 0.0
        for variant in variants:
            cumulative += variant.traffic_allocation
            if random_value < cumulative:
                return variant.variant_id

        # Fallback to last variant
        return variants[-1].variant_id

    def assign_variant(
        self,
        user_id: str,
        experiment_id: str,
        session_id: Optional[str] = None,
        user_attributes: Optional[Dict] = None
    ) -> ExperimentAssignment:
        """
        Assign user to a variant.

        Args:
            user_id: User ID
            experiment_id: Experiment ID
            session_id: Optional session ID
            user_attributes: Optional user attributes for targeting

        Returns:
            ExperimentAssignment
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment {experiment_id} is not running")

        # Check if user already assigned
        existing = [
            a for a in self.assignments
            if a.user_id == user_id and a.experiment_id == experiment_id
        ]
        if existing:
            return existing[0]

        # Check targeting rules
        if experiment.targeting_rules and user_attributes:
            if not self._matches_targeting(user_attributes, experiment.targeting_rules):
                # User doesn't match targeting, use control
                variant_id = [v for v in experiment.variants if v.variant_type == VariantType.CONTROL][0].variant_id
            else:
                variant_id = self._hash_user_to_variant(
                    user_id, experiment_id, experiment.variants
                )
        else:
            variant_id = self._hash_user_to_variant(
                user_id, experiment_id, experiment.variants
            )

        assignment = ExperimentAssignment(
            user_id=user_id,
            experiment_id=experiment_id,
            variant_id=variant_id,
            session_id=session_id
        )

        self.assignments.append(assignment)
        return assignment

    def _matches_targeting(self, user_attrs: Dict, rules: Dict) -> bool:
        """Check if user matches targeting rules."""
        for key, value in rules.items():
            if key not in user_attrs:
                return False
            if isinstance(value, list):
                if user_attrs[key] not in value:
                    return False
            elif user_attrs[key] != value:
                return False
        return True

    def track_event(
        self,
        user_id: str,
        experiment_id: str,
        metric_name: str,
        value: float = 1.0,
        metadata: Optional[Dict] = None
    ):
        """
        Track an event for experiment analysis.

        Args:
            user_id: User ID
            experiment_id: Experiment ID
            metric_name: Name of metric
            value: Value of metric
            metadata: Optional metadata
        """
        # Get user's variant assignment
        assignment = [
            a for a in self.assignments
            if a.user_id == user_id and a.experiment_id == experiment_id
        ]

        if not assignment:
            # User not in experiment, skip
            return

        variant_id = assignment[0].variant_id

        event = ExperimentEvent(
            user_id=user_id,
            experiment_id=experiment_id,
            variant_id=variant_id,
            metric_name=metric_name,
            value=value,
            metadata=metadata or {}
        )

        self.events.append(event)

    def get_results(
        self,
        experiment_id: str,
        confidence_level: float = 0.95
    ) -> ExperimentResult:
        """
        Get experiment results with statistical analysis.

        Args:
            experiment_id: Experiment ID
            confidence_level: Confidence level for significance testing

        Returns:
            ExperimentResult
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Get events for this experiment
        exp_events = [e for e in self.events if e.experiment_id == experiment_id]

        # Calculate results per variant
        variant_results = {}
        sample_sizes = {}

        for variant in experiment.variants:
            variant_events = [e for e in exp_events if e.variant_id == variant.variant_id]

            # Get unique users in this variant
            variant_assignments = [
                a for a in self.assignments
                if a.experiment_id == experiment_id and a.variant_id == variant.variant_id
            ]
            sample_sizes[variant.variant_id] = len(variant_assignments)

            # Calculate metrics
            metrics = {}
            for metric in experiment.metrics:
                metric_events = [
                    e for e in variant_events
                    if e.metric_name == metric.metric_name
                ]

                if metric.metric_type == "count":
                    metrics[metric.metric_name] = len(metric_events)
                elif metric.metric_type == "rate":
                    # Calculate conversion rate
                    unique_users = len(set([e.user_id for e in metric_events]))
                    metrics[metric.metric_name] = (
                        unique_users / sample_sizes[variant.variant_id]
                        if sample_sizes[variant.variant_id] > 0 else 0.0
                    )
                elif metric.metric_type == "revenue":
                    metrics[metric.metric_name] = sum(e.value for e in metric_events)
                elif metric.metric_type == "time":
                    metrics[metric.metric_name] = (
                        np.mean([e.value for e in metric_events])
                        if metric_events else 0.0
                    )

            variant_results[variant.variant_id] = metrics

        # Statistical significance testing
        statistical_significance = {}

        # Find control and treatment variants
        control_variant = [v for v in experiment.variants if v.variant_type == VariantType.CONTROL]
        treatment_variants = [v for v in experiment.variants if v.variant_type == VariantType.TREATMENT]

        if control_variant and treatment_variants:
            control_id = control_variant[0].variant_id

            for metric in experiment.metrics:
                if metric.metric_type == "rate":
                    # Use proportion test
                    control_events = [
                        e for e in exp_events
                        if e.variant_id == control_id and e.metric_name == metric.metric_name
                    ]
                    control_conversions = len(set([e.user_id for e in control_events]))
                    control_sample = sample_sizes[control_id]

                    # Test against first treatment
                    treatment_id = treatment_variants[0].variant_id
                    treatment_events = [
                        e for e in exp_events
                        if e.variant_id == treatment_id and e.metric_name == metric.metric_name
                    ]
                    treatment_conversions = len(set([e.user_id for e in treatment_events]))
                    treatment_sample = sample_sizes[treatment_id]

                    if control_sample > 0 and treatment_sample > 0:
                        # Two-proportion z-test
                        p_value = self._proportion_test(
                            control_conversions, control_sample,
                            treatment_conversions, treatment_sample
                        )
                        statistical_significance[metric.metric_name] = p_value < (1 - confidence_level)
                    else:
                        statistical_significance[metric.metric_name] = False

        # Generate recommendation
        recommendation = self._generate_recommendation(
            variant_results, statistical_significance, experiment
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            variant_results=variant_results,
            statistical_significance=statistical_significance,
            confidence_level=confidence_level,
            sample_sizes=sample_sizes,
            recommendation=recommendation
        )

    def _proportion_test(
        self,
        control_conversions: int,
        control_sample: int,
        treatment_conversions: int,
        treatment_sample: int
    ) -> float:
        """Perform two-proportion z-test."""
        p1 = control_conversions / control_sample
        p2 = treatment_conversions / treatment_sample

        p_pool = (control_conversions + treatment_conversions) / (control_sample + treatment_sample)

        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_sample + 1/treatment_sample))

        if se == 0:
            return 1.0

        z = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return p_value

    def _generate_recommendation(
        self,
        variant_results: Dict[str, Dict[str, float]],
        statistical_significance: Dict[str, bool],
        experiment: Experiment
    ) -> str:
        """Generate recommendation based on results."""
        # Find primary metric
        primary_metrics = [m for m in experiment.metrics if m.primary]

        if not primary_metrics:
            return "No primary metric defined. Review results manually."

        primary_metric = primary_metrics[0].metric_name

        # Check if significant
        if primary_metric not in statistical_significance:
            return "Insufficient data for statistical analysis."

        if not statistical_significance[primary_metric]:
            return "No statistically significant difference detected. Consider running longer or using control."

        # Find best performing variant
        best_variant_id = max(
            variant_results.keys(),
            key=lambda v: variant_results[v].get(primary_metric, 0)
        )

        best_variant = [v for v in experiment.variants if v.variant_id == best_variant_id][0]

        improvement = (
            variant_results[best_variant_id][primary_metric] -
            min(variant_results[v][primary_metric] for v in variant_results.keys())
        )

        return f"Ship {best_variant.name}! Improvement of {improvement:.2%} in {primary_metric} is statistically significant."

    def get_experiment_summary(self, experiment_id: str) -> Dict:
        """Get summary of experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        assignments = [a for a in self.assignments if a.experiment_id == experiment_id]
        events = [e for e in self.events if e.experiment_id == experiment_id]

        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "total_users": len(set([a.user_id for a in assignments])),
            "total_events": len(events),
            "variants": {
                v.variant_id: {
                    "name": v.name,
                    "type": v.variant_type,
                    "traffic_allocation": v.traffic_allocation,
                    "assigned_users": len([a for a in assignments if a.variant_id == v.variant_id])
                }
                for v in experiment.variants
            },
            "duration_hours": (
                (datetime.utcnow() - experiment.start_date).total_seconds() / 3600
                if experiment.start_date else 0
            )
        }
