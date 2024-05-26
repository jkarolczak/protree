from __future__ import annotations

from typing import Type, Literal

from river.base import DriftDetector
from river.forest import ARFClassifier
from sklearn.ensemble import RandomForestClassifier

from protree import TPrototypes
from protree.explainers import TExplainer, APete


class Ancient(DriftDetector):
    """Algorithm for New Concept Identification and Explanation in Tree-based models

    """

    def __init__(self, model: RandomForestClassifier | ARFClassifier, prototype_selector: Type[TExplainer] = APete,
                 prototype_selector_kwargs: dict = {}, clock: int = 32, alpha: float = 1.5,
                 measure: Literal["mutual_info", "centroid_displacement", "minimal_distance"] = "centroid_displacement",
                 strategy: Literal["class", "total"] = "class", window_length: int = 200,
                 cold_start: int = 1000) -> None:
        super().__init__()
        self.alpha = alpha
        self.clock = clock
        self.cold_start = cold_start
        self.measure = measure
        self.model = model
        self.prototype_selector = prototype_selector
        self.prototype_selector_kwargs = prototype_selector_kwargs
        self.strategy = strategy
        self.window_length = window_length

        self.x_window = []
        self.y_window = []

        self._iter_counter = 0
        self._check_counter = 0
        self._in_drift = False

    def update(self, x, y) -> None:
        self._update_x_window(x)
        self._update_y_window(y)
        self._update_counter()
        self._in_drift = False
        if not self._check_counter and (self._iter_counter > self.cold_start) and len(self.x_window) == self.window_length:
            self._detect_drift()

    def _update_x_window(self, x) -> None:
        self.x_window.append(x)
        if len(self.x_window) > self.window_length:
            self.x_window.pop(0)

    def _update_y_window(self, y) -> None:
        self.y_window.append(y)
        if len(self.y_window) > self.window_length:
            self.y_window.pop(0)

    def _update_counter(self):
        self._check_counter = (self._check_counter + 1) % self.clock
        self._iter_counter += 1

    def _reset(self):
        super()._reset()
        self.x_window = []

    def _find_prototypes(self) -> tuple[TPrototypes, TPrototypes]:
        a_x = self.x_window[:int(len(self.x_window) / 2)]
        a_y = self.y_window[:int(len(self.y_window) / 2)]
        b_x = self.x_window[int(len(self.x_window) / 2):]
        b_y = self.y_window[int(len(self.y_window) / 2):]

        explainer_a: TExplainer = self.prototype_selector(self.model, **self.prototype_selector_kwargs)
        prototypes_a = explainer_a.select_prototypes(a_x, a_y)

        explainer_b: TExplainer = self.prototype_selector(self.model, **self.prototype_selector_kwargs)
        prototypes_b = explainer_b.select_prototypes(b_x, b_y)

        return prototypes_a, prototypes_b

    def _detect_drift(self) -> None:
        if self.measure == "mutual_info":
            self._detect_mutual_info()
        elif self.measure == "minimal_distance":
            if self.strategy == "class":
                self._detect_minimal_distance_class()
            elif self.strategy == "total":
                self._detect_minimal_distance_total()
        elif self.measure == "centroid_displacement":
            if self.strategy == "class":
                self._detect_centroid_displacement_class()
            elif self.strategy == "total":
                self._detect_centroid_displacement_total()

    def _detect_mutual_info(self) -> None:
        from protree.metrics.compare import mutual_information

        prototypes_a, prototypes_b = self._find_prototypes()

        mutual_info = mutual_information(prototypes_a, prototypes_b, self.x_window)

        if mutual_info < self.alpha:
            self._in_drift = True

    def _detect_minimal_distance_class(self) -> None:
        from protree.metrics.compare import classwise_mean_minimal_distance

        prototypes_a, prototypes_b = self._find_prototypes()
        minimal_distance = classwise_mean_minimal_distance(prototypes_a, prototypes_b)

        for label in minimal_distance:
            if minimal_distance[label] > self.alpha:
                self._in_drift = True
                break

    def _detect_minimal_distance_total(self) -> None:
        from protree.metrics.compare import mean_minimal_distance

        prototypes_a, prototypes_b = self._find_prototypes()
        minimal_distance = mean_minimal_distance(prototypes_a, prototypes_b)

        if minimal_distance > self.alpha:
            self._in_drift = True

    def _detect_centroid_displacement_total(self) -> None:
        from protree.metrics.compare import mean_centroid_displacement

        prototypes_a, prototypes_b = self._find_prototypes()
        displacement = mean_centroid_displacement(prototypes_a, prototypes_b)

        if displacement > self.alpha:
            self._in_drift = True

    def _detect_centroid_displacement_class(self) -> None:
        from protree.metrics.compare import centroids_displacements

        prototypes_a, prototypes_b = self._find_prototypes()
        displacement = centroids_displacements(prototypes_a, prototypes_b)

        for label in displacement:
            if displacement[label] > self.alpha:
                self._in_drift = True
                break

    @property
    def drift_detected(self) -> bool:
        return self._in_drift
