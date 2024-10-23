from __future__ import annotations

from copy import deepcopy
from typing import Type, Literal, Callable

from river.base import DriftDetector
from river.forest import ARFClassifier
from sklearn.ensemble import RandomForestClassifier

from protree import TPrototypes, TDataBatch, TTarget
from protree.explainers import TExplainer, APete


class ProtoBDrift(DriftDetector):
    """Prototypes-based drift detector for block data streams.

    """

    def __init__(self, model: RandomForestClassifier | ARFClassifier, prototype_selector: Type[TExplainer] = APete,
                 prototype_selector_kwargs: dict | None = None,
                 measure: Literal[
                     "mutual_information", "rand_index", "completeness", "fowlkes_mallows", "centroid_displacement",
                     "minimal_distance"] = "centroid_displacement", strategy: Literal["class", "total"] = "total",
                 distance: Literal["l2", "tree"] = "l2", assign_to: Literal["class", "prototype"] = "prototype",
                 cold_start: int = 20, const: float = 0.33) -> None:
        super().__init__()
        self.assign_to = assign_to
        self.cold_start = cold_start
        self.const = const
        self.distance = distance
        self.measure = measure
        self.model = model
        self.prototype_selector = prototype_selector
        self.prototype_selector_kwargs = prototype_selector_kwargs or {}
        self.strategy = strategy

        self._metric_significance_max: int | dict[int | str, float] | None = None
        self._metric_significance_min: int | dict[int | str, float] | None = None

        self._direction: str | None = None
        self._iter_counter: int = 0
        self._drift_detected: bool = False

        self._init_state()

    def _init_state(self) -> None:
        self.x_blocks: tuple[TDataBatch, TDataBatch] = (None, None)
        self.y_blocks: tuple[TTarget, TTarget] = (None, None)

        self.explainers: tuple[TExplainer | None, TExplainer | None] = (None, None)

        self.prototypes: tuple[TPrototypes, TPrototypes] = ({}, {})

    def update(self, x: TDataBatch, y: TTarget) -> None:
        self._update_block(x, y)
        self._drift_detected = False
        self._update_explainers()
        self._find_prototypes()
        if self._iter_counter >= 1:
            metric = self._compute_metric()

            if (self.cold_start / 2) <= self._iter_counter < self.cold_start:
                self._update_metric_stats(metric)
            elif self.cold_start <= self._iter_counter:
                self._test(metric)
        self._iter_counter += 1

    def _update_block(self, x: TDataBatch, y: TTarget) -> None:
        self.x_blocks = (self.x_blocks[1], x)
        self.y_blocks = (self.y_blocks[1], y)

    def _update_explainers(self) -> None:
        self.explainers = (
            deepcopy(self.explainers[1]),
            self.prototype_selector(deepcopy(self.model), **self.prototype_selector_kwargs)
        )

    def _find_prototypes(self) -> None:
        self.prototypes = (
            deepcopy(self.prototypes[1]),
            self.explainers[1].select_prototypes(self.x_blocks[1], self.y_blocks[1])
        )

    def _update_metric_stats(self, metric: float | dict[str | int, float]) -> None:
        if self._metric_significance_min is None:
            if isinstance(metric, float):
                self._metric_significance_min = metric
                self._metric_significance_max = metric
            elif isinstance(metric, dict):
                self._metric_significance_min = {}
                self._metric_significance_max = {}
                for key in metric:
                    self._metric_significance_min[key] = metric
                    self._metric_significance_max[key] = metric

        elif isinstance(self._metric_significance_min, float):
            self._metric_significance_min = min(self._metric_significance_min, metric)
            self._metric_significance_max = max(self._metric_significance_max, metric)

        elif isinstance(self._metric_significance_min, dict):
            for key in metric:
                self._metric_significance_min[key] = min(self._metric_significance_min[key], metric[key])
                self._metric_significance_max[key] = max(self._metric_significance_max[key], metric[key])

    def _reset(self):
        super()._reset()
        self._init_state()

    def _test_value(self, value: float, key: int | str | None = None) -> bool:
        if key is not None:
            interval = self._metric_significance_max[key] - self._metric_significance_min[key]
            if self._direction == "increase":
                return value > self._metric_significance_max[key] + self.const * interval
            else:
                return value < self._metric_significance_min[key] - self.const * interval

        interval = self._metric_significance_max - self._metric_significance_min

        if self._direction == "increase":
            return value > self._metric_significance_max + self.const * interval
        else:
            return value < self._metric_significance_min - self.const * interval

    def _test(self, metric: float | dict[int | str, float]) -> None:
        # strategy: class
        if isinstance(metric, dict):
            for key in metric:
                if self._test_value(metric[key], key=key):
                    self._drift_detected = True
                    break

        # strategy: total
        elif isinstance(metric, float):
            if self._test_value(metric, key=None):
                self._drift_detected = True

    def _compute_metric(self) -> float | dict[int | str, float]:
        if self.measure in ["mutual_information", "rand_index", "completeness", "fowlkes_mallows"]:
            import protree.metrics.compare

            self._direction = "decrease"

            metric = getattr(protree.metrics.compare, self.measure)
            return self._compute_cluster_metric(metric)

        elif self.measure == "swap_delta":
            from protree.metrics.compare import swap_delta

            self._direction = "increase"
            if self.distance == "tree":
                return swap_delta(*self.prototypes, self.x_blocks[1], self.y_blocks[1], explainer=self.explainers[1])
            return swap_delta(*self.prototypes, self.x_blocks[1], self.y_blocks[1])

        elif self.measure == "minimal_distance":
            self._direction = "increase"

            if self.strategy == "class":
                from protree.metrics.compare import classwise_mean_minimal_distance

                return self._compute_spatial_metric(classwise_mean_minimal_distance)

            elif self.strategy == "total":
                from protree.metrics.compare import mean_minimal_distance

                return self._compute_spatial_metric(mean_minimal_distance)

        elif self.measure == "centroid_displacement":
            self._direction = "increase"

            if self.strategy == "class":
                from protree.metrics.compare import centroids_displacements

                return self._compute_spatial_metric(centroids_displacements)

            elif self.strategy == "total":
                from protree.metrics.compare import mean_centroid_displacement

                return self._compute_spatial_metric(mean_centroid_displacement)

    def _build_cluster_kwargs(self) -> dict:
        kwargs = {
            "a": self.prototypes[0],
            "b": self.prototypes[1],
            "x": self.x_blocks[1],
            "assign_to": self.assign_to
        }

        if self.distance == "tree":
            kwargs["explainer_a"] = self.explainers[0]
            kwargs["explainer_b"] = self.explainers[1]

        return kwargs

    def _compute_cluster_metric(self, metric: Callable[..., float]) -> float | dict[int | str, float]:
        return metric(**self._build_cluster_kwargs())

    def _compute_spatial_metric(self, metric: Callable[..., float | dict[int | str, float]]) -> float | dict[int | str, float]:
        return metric(*self.prototypes)
