from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias, Literal

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from protree import TDataBatch, TTarget
from protree.meta import RANDOM_SEED

TStreamGenerator: TypeAlias = Literal["sine", "plane", "random_tree", "make_classification"]


class StreamGeneratorFactory:
    @staticmethod
    def create(name: TStreamGenerator, drift_position: int | list[int] = 500, drift_width: int = 5, seed: int = RANDOM_SEED
               ) -> IStreamGenerator:
        name = ''.join([n.capitalize() for n in name.split('_')])
        return globals()[f"{name}"](drift_position=drift_position, drift_width=drift_width, seed=seed)


class IStreamGenerator(ABC):
    def __init__(self, drift_position: int | list[int] = 500, drift_width: int = 0, seed: int = 42) -> None:
        self.drift_position = drift_position if isinstance(drift_position, (list, tuple)) else [drift_position]
        self.drift_width = drift_width
        self.seed = seed
        self._iter_counter = 0
        self._iter_drift_remaining = 0
        np.random.seed(seed)

    @abstractmethod
    def take(self, n: int) -> list[tuple[TDataBatch, TTarget]]:
        pass

    def __iter__(self) -> IStreamGenerator:
        return self

    def __next__(self) -> tuple[dict[str, float], int]:
        return self.take(1)[0]


class Sine(IStreamGenerator):
    def __init__(self, drift_position: int | list[int] = 500, drift_width: int = 1, seed: int = 42,
                 informative_attrs: tuple[int, int] = (3, 2)) -> None:
        super().__init__(drift_position, drift_width, seed)
        self.informative_attrs_0 = informative_attrs
        self.informative_attrs_1 = (None, None)

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            x_rand = np.random.uniform(0, 1)
            x = {f"x{i}": getattr(self, f"x{i}")(x_rand) for i in range(6)}
            y = self.y(x)
            result.append((x, y))
            self._iter_counter += 1
        return result

    def _generate_new_informative_attrs(self, old_informative_attrs: tuple[int, int]) -> tuple[int, int]:
        if ((new_attrs := np.random.choice([0, 1, 2, 3], 2, replace=False)) == old_informative_attrs).all():
            return self._generate_new_informative_attrs(old_informative_attrs)
        return new_attrs

    def update_drift(self) -> None:
        if self._iter_counter in self.drift_position:
            self.informative_attrs_1 = self._generate_new_informative_attrs(self.informative_attrs_0)
            self._iter_drift_remaining = self.drift_width
        if self._iter_drift_remaining == 1:
            self.informative_attrs_0 = self.informative_attrs_1
            self.informative_attrs_1 = (None, None)
        self._iter_drift_remaining -= 1

    def y(self, x: dict[str, float]) -> int:
        if self._iter_drift_remaining > 0 and np.random.uniform(0, 1) < 1 / self._iter_drift_remaining:
            return int(x[f"x{self.informative_attrs_1[0]}"] > x[f"x{self.informative_attrs_1[1]}"])
        return int(x[f"x{self.informative_attrs_0[0]}"] > x[f"x{self.informative_attrs_0[1]}"])

    @property
    def informative_attrs(self) -> tuple[int, int]:
        return self.informative_attrs_0

    def x0(self, x: float) -> float:
        return np.sin(np.pi * x).item()

    def x1(self, x: float) -> float:
        return (np.sin(2 * np.pi * x + np.pi / 2).item() + 1) / 2

    def x2(self, x: float) -> float:
        return np.sin(2 * np.pi / 3 * x).item()

    def x3(self, x: float) -> float:
        return (np.sin(3 * np.pi * x + np.pi / 3).item() + 1) / 2

    def x4(self, x: float) -> float:
        return np.random.uniform(0, 1)

    def x5(self, x: float) -> float:
        return np.clip(np.random.randn(1), 0, 1).item()


class Plane(IStreamGenerator):
    def __init__(self, drift_position: int | list[int] = 500, drift_width: int = 1, seed: int = 42,
                 decision_func: int = 1) -> None:
        super().__init__(drift_position, drift_width, seed)
        self.decision_func_0 = decision_func
        self.decision_func_1 = None

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            x = {f"x{i}": np.random.uniform(0, 1) for i in range(3)}
            y = self.y(x)
            result.append((x, y))
            self._iter_counter += 1
        return result

    def _generate_new_decision_function(self, old_decision_func: int) -> int:
        if (new_func := np.random.choice([0, 1, 2, 3])) == old_decision_func:
            return self._generate_new_decision_function(old_decision_func)
        return new_func

    def update_drift(self) -> None:
        if self._iter_counter in self.drift_position:
            self.decision_func_1 = self._generate_new_decision_function(self.decision_func_0)
            self._iter_drift_remaining = self.drift_width
        if self._iter_drift_remaining == 1:
            self.decision_func_0 = self.decision_func_1
            self.decision_func_1 = None
        self._iter_drift_remaining -= 1

    def y(self, x: dict[str, float]) -> int:
        if self._iter_drift_remaining > 0 and np.random.uniform(0, 1) < 1 / self._iter_drift_remaining:
            return getattr(self, f"y{self.decision_func_1}")(x)
        return getattr(self, f"y{self.decision_func_0}")(x)

    def y0(self, x: dict[str, float]) -> int:
        return int(-x["x0"] + x["x1"] > 1)

    def y1(self, x: dict[str, float]) -> int:
        return int(-3 * x["x0"] + x["x1"] > -1)

    def y2(self, x: dict[str, float]) -> int:
        return int(-3 * x["x0"] - x["x1"] < -1)

    def y3(self, x: dict[str, float]) -> int:
        return int(-x["x0"] + 3 * x["x1"] < 1)


class RandomTree(IStreamGenerator):
    def __init__(self, drift_position: int | list[int] = 500, drift_width: int = 1, seed: int = 42,
                 n_informative: int = 4, n_redundant: int = 2) -> None:
        super().__init__(drift_position, drift_width, seed)
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self._tree = None
        self._flip = False

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            x = {f"x{i}": np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1).item() for i in
                 range(self.n_informative + self.n_redundant)}
            y = self.y(x)
            result.append((x, y))
            self._iter_counter += 1
        return result

    def update_drift(self) -> None:
        if self._iter_counter in self.drift_position:
            self._iter_drift_remaining = self.drift_width
        if self._iter_drift_remaining == 1:
            self._flip = not self._flip
        self._iter_drift_remaining -= 1

    def _grow_tree(self) -> None:
        self._tree = DecisionTreeClassifier().fit(np.random.rand(80, self.n_informative), np.random.randint(0, 2, 80))

    def y(self, x: dict[str, float]) -> int:
        if not self._tree:
            self._grow_tree()
        tree_prediction = self._tree.predict(np.array([list(x.values())[:self.n_informative]]))[0]
        return int(self._flip != tree_prediction)


class MakeClassification(IStreamGenerator):
    def __init__(self, drift_position: int | list[int] = 500, drift_width: int = 1, seed: int = 42,
                 n_informative: int = 5, n_redundant: int = 0, stream_length: int = 30000) -> None:
        super().__init__(drift_position, drift_width, seed)
        self.stream = make_classification(n_features=n_informative + n_redundant, n_informative=n_informative,
                                          n_redundant=n_redundant, n_clusters_per_class=4,
                                          n_classes=2, n_samples=stream_length, random_state=seed)
        self._flip = False

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            x = {f"x{idx}": np.clip(val / 10 + 0.5, 0, 1).item() for idx, val in enumerate(self.stream[0][self._iter_counter])}
            y = int(self.stream[1][self._iter_counter] != self._flip)
            result.append((x, y))
            self._iter_counter += 1
        return result

    def update_drift(self):
        if self._iter_counter in self.drift_position:
            self._iter_drift_remaining = self.drift_width
        if self._iter_drift_remaining == 1:
            self._flip = not self._flip
        self._iter_drift_remaining -= 1
