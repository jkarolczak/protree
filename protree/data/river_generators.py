from __future__ import annotations

import numpy as np
from river.datasets import synth

from protree.data.stream_generators import IStreamGenerator


class Agrawal(IStreamGenerator):
    """Agrawal synthetic dataset, based on the River library implementation:
    https://riverml.xyz/latest/api/datasets/synth/Agrawal/

    The simulated drift is a transition between two classification functions available in the dataset.

    :param drift_position: The position of the drift.
    :param drift_duration: The width of the drift.
    :param seed: Random seed.
    """

    def __init__(self, drift_position: int | list[int] = 500, drift_duration: int = 0, classification_function: int = 6,
                 seed: int = 42) -> None:
        super().__init__(drift_position, drift_duration, seed)
        self.drift_position = drift_position if isinstance(drift_position, (list, tuple)) else [drift_position]
        self.drift_duration = drift_duration
        self.seed = seed
        self._iter_counter = 0
        self._iter_drift_remaining = 0
        self._data_streams = (synth.Agrawal(classification_function=6, balance_classes=True, seed=seed).__iter__(), None)
        np.random.seed(seed)

    def update_drift(self) -> None:
        if self._iter_counter in self.drift_position:
            self._data_streams = (
                self._data_streams[0],
                synth.Agrawal(classification_function=np.random.randint(0, 9), seed=self.seed).__iter__()
            )
            self._iter_drift_remaining = self.drift_duration
        if self._iter_drift_remaining == 1:
            self._data_streams = self._data_streams[1], None
        self._iter_drift_remaining -= 1

    def _normalise(self, x: dict[str, float]) -> dict[str, float]:
        return {
            "salary": (x["salary"] - 20000) / 130000,
            "commission": 0 if x["commission"] == 0 else (x["commission"] - 10000) / 65000,
            "age": (x["age"] - 20) / 60,
            "elevel": x["elevel"] / 4,
            "zipcode": x["zipcode"] / 8,
            "hvalue": x["hvalue"] / 800000,
            "hyears": x["hyears"] / 30,
            "loan": x["loan"] / 500000,
        }

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            if self._iter_drift_remaining > 0 and np.random.uniform(0, 1) < self.drift_factor():
                x, y = next(self._data_streams[1])
            else:
                x, y = next(self._data_streams[0])
            x = self._normalise(x)
            result.append((x, y))
        return result


class SEA(IStreamGenerator):
    """SEA synthetic dataset, based on the River library implementation:
    https://riverml.xyz/latest/api/datasets/synth/SEA/

    The simulated drift is a transition between two classification functions available in the dataset.

    :param drift_position: The position of the drift.
    :param drift_duration: The width of the drift.
    :param seed: Random seed.
    """

    def __init__(self, drift_position: int | list[int] = 500, drift_duration: int = 0, seed: int = 42) -> None:
        super().__init__(drift_position, drift_duration, seed)
        self.drift_position = drift_position if isinstance(drift_position, (list, tuple)) else [drift_position]
        self.drift_duration = drift_duration
        self.seed = seed
        self._iter_counter = 0
        self._iter_drift_remaining = 0
        self._data_streams = (synth.SEA(variant=3, seed=seed).__iter__(), None)
        np.random.seed(seed)

    def update_drift(self) -> None:
        if self._iter_counter in self.drift_position:
            self._data_streams = (
                self._data_streams[0],
                synth.SEA(variant=np.random.randint(0, 4), seed=self.seed).__iter__()
            )
            self._iter_drift_remaining = self.drift_duration
        if self._iter_drift_remaining == 1:
            self._data_streams = self._data_streams[1], None
        self._iter_drift_remaining -= 1

    def _normalise(self, x: dict[str, float]) -> dict[str, float]:
        return {k: v / 10 for k, v in x.items()}

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            if self._iter_drift_remaining > 0 and np.random.uniform(0, 1) < self.drift_factor():
                x, y = next(self._data_streams[1])
            else:
                x, y = next(self._data_streams[0])
            x = self._normalise(x)
            result.append((x, y))
        return result


class RBF(IStreamGenerator):
    def __init__(self, drift_position: int | list[int] = 500, drift_duration: int = 1, seed: int = 42,
                 n_informative: int = 5, n_centroids: int = 11) -> None:
        super().__init__(drift_position, drift_duration, seed)
        self._data_stream = synth.RandomRBF(seed_model=seed, seed_sample=seed, n_features=n_informative,
                                            n_centroids=n_centroids).__iter__()
        self._flip = False

    def _permute_y(self, y: int) -> int:
        map_ = {0: 2, 1: 3, 2: 0, 3: 1}
        return map_[y]

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            x, y = next(self._data_stream)
            if self._flip:
                y = self._permute_y(y)
            if self._iter_drift_remaining > 0 and np.random.uniform(0, 1) < self.drift_factor():
                y = self._permute_y(y)

            result.append((x, y))
            self._iter_counter += 1
        return result

    def update_drift(self):
        if self._iter_counter in self.drift_position:
            self._iter_drift_remaining = self.drift_duration
        if self._iter_drift_remaining == 1:
            self._flip = not self._flip
        self._iter_drift_remaining -= 1
