from __future__ import annotations

from enum import Enum
from typing import Type

from river.base import DriftDetector

from protree import TPrototypes, TDataBatch
from protree.explainers import TExplainer, APete


class Ancient(DriftDetector):
    """Algorithm for New Concept Identification and Explanation in Tree-based models

    """

    class Condition(Enum):
        TOTAL_CHANGE = 1
        ONE_CLASS = 2
        ALL_CLASSES = 3

    def __init__(self, prototype_selector: Type[TExplainer] = APete, prototype_selector_kwargs: dict = {},
                 window_size: int = 200, alpha: float = 0.10, condition: Condition = Condition.TOTAL_CHANGE,
                 chunks_split: float = 0.5) -> None:
        super().__init__()
        self.prototype_selector = prototype_selector
        self.prototype_selector_kwargs = prototype_selector_kwargs
        self.window_size = window_size
        self.alpha = alpha
        self.condition = condition
        self._drift_detection_function = self.select_detection_function()
        self.window = []
        self.chunks_split = chunks_split
        self._in_drift = False
        self._in_warning = False

    def update(self, x) -> Ancient:
        self.window.append(x)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        self._detect_drift()
        return self.in_drift, in_warning

    def _reset(self):
        super()._reset()
        self.window = []

    def select_detection_function(self) -> callable:
        match self.condition:
            case Ancient.Condition.TOTAL_CHANGE:
                return self._if_total_change_is_significant
            case Ancient.Condition.ONE_CLASS:
                return self._if_at_least_one_class_changed_significantly
            case Ancient.Condition.ALL_CLASSES:
                return self._if_each_class_changed_significantly

    def _if_total_change_is_significant(self, a: TPrototypes, b: TPrototypes) -> bool:
        pass

    def _if_at_least_one_class_changed_significantly(self, a: TPrototypes, b: TPrototypes) -> bool:
        return any(self._if_class_changed_significantly(a[cls], b[cls]) for cls in set(a.keys()).union(b.keys()))

    def _if_each_class_changed_significantly(self, a: TPrototypes, b: TPrototypes) -> bool:
        return all(self._if_class_changed_significantly(a[cls], b[cls]) for cls in set(a.keys()).union(b.keys()))

    def _if_class_changed_significantly(self, a: TDataBatch, b: TDataBatch) -> float:
        pass

    def _detect_drift(self):
        a = self.window[:int(self.chunks_split * len(self.window))]
        b = self.window[int(self.chunks_split * len(self.window)):]

        explainer_a: TExplainer = self.prototype_selector(**self.prototype_selector_kwargs)
        prototypes_a = explainer_a.select_prototypes(a)

        explainer_b: TExplainer = self.prototype_selector(**self.prototype_selector_kwargs)
        prototypes_b = explainer_b.select_prototypes(b)

        return self._drift_detection_function(prototypes_a, prototypes_b)

    @property
    def drift_detected(self) -> bool:
        return self._in_drift
