import warnings
from abc import ABC
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as _SKLearnRandomForestClassifier

from explainers.utils import parse_input, _type_to_np_dtype
from metrics.classification import balanced_accuracy


class IModelAdapter(ABC):
    @property
    def n_estimators(self) -> int:
        pass

    def get_leave_indices(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        pass

    def get_model_predictions(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        pass


class SKLearnAdapter(IModelAdapter):
    def __init__(self, model: _SKLearnRandomForestClassifier) -> None:
        self.model = model

    @property
    def n_estimators(self) -> int:
        return self.model.n_estimators

    def get_leave_indices(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.apply(parse_input(x))

    def get_model_predictions(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict(parse_input(x))


class ModelAdapterBuilder:
    def __init__(self, model: _SKLearnRandomForestClassifier) -> None:
        self.model = model

    def __call__(self, *args, **kwargs) -> IModelAdapter:
        if isinstance(self.model, _SKLearnRandomForestClassifier):
            return SKLearnAdapter(model=self.model)
        raise ValueError("Unsupported model class.")


class IExplainer(ABC):
    def __init__(self, model: _SKLearnRandomForestClassifier, *args, **kwargs) -> None:
        self.model = ModelAdapterBuilder(model)()

    def par_similarity(self, x1: np.ndarray | pd.Series, x2: np.ndarray | pd.Series) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            nodes1 = self.model.get_leave_indices(parse_input(x1))
            nodes2 = self.model.get_leave_indices(parse_input(x2))

        return sum((nodes1 == nodes2)[0]) / self.model.n_estimators

    def similarity_matrix(self, df: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nodes = self.model.get_leave_indices(df)

        shape = df.shape[0]
        matrix = np.zeros((shape, shape), dtype=float)
        for i in range(shape):
            matrix[i] = (nodes[i] == nodes).sum(axis=1)
        matrix /= self.model.n_estimators
        return matrix

    def distance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return 1 - self.similarity_matrix(df)

    def _create_distance_matrices(self, x: pd.DataFrame, y: pd.DataFrame, classes: Iterable[int | str]
                                  ) -> dict[int | str, np.ndarray]:
        distances = {cls: None for cls in classes}
        for cls in classes:
            class_x = x[(y == cls).any(axis=1)]
            distances[cls] = self.distance_matrix(class_x)
        return distances

    def get_prototypes_predictions(self, x: pd.DataFrame, prototypes: dict[int | str, pd.DataFrame]) -> np.ndarray:
        predictions = (np.ones((len(x), 1)) * (-1)).astype(_type_to_np_dtype(prototypes))
        similarity = np.zeros((len(x)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_nodes = self.model.get_leave_indices(x)

        for cls in prototypes:
            for _, prototype in prototypes[cls].iterrows():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prototype_leaves = self.model.get_leave_indices([prototype])
                prototype_similarity = (prototype_leaves == x_nodes).sum(axis=1) / self.model.n_estimators
                mask = prototype_similarity > similarity
                predictions[mask] = cls
                similarity[mask] = prototype_similarity[mask]
        return predictions

    @staticmethod
    def _score(y: pd.DataFrame, y_hat: np.ndarray) -> float:
        y = y.values.squeeze(1)
        return balanced_accuracy(y, y_hat)

    def score_with_prototypes(self, x: pd.DataFrame, y: pd.DataFrame, prototypes: dict[int | str, pd.DataFrame]) -> float:
        y_hat = self.get_prototypes_predictions(x, prototypes)
        return IExplainer._score(y, y_hat)

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        y_hat = self.model.get_model_predictions(x)
        return IExplainer._score(y, y_hat)


class G_KM(IExplainer):
    def __init__(self, model: _SKLearnRandomForestClassifier, n_prototypes: int = 3, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.n_prototypes = n_prototypes

    def _find_classwise_prototype(self, matrix: np.ndarray, prototypes: list[int]) -> int:
        if not prototypes:
            idx = np.argmin(matrix.sum(axis=1))
            return idx

        mask = np.in1d(range(matrix.shape[0]), prototypes)
        current_partial_distances = matrix[mask].min(axis=0)
        candidates = matrix[~mask]
        candidate_distances = np.minimum(candidates, current_partial_distances).sum(axis=1)
        original_idx = np.where(~mask)[0][np.argmin(candidate_distances)]
        return original_idx

    def _find_single_class_prototypes(self, matrix: np.ndarray) -> list[int]:
        prototypes = []
        for _ in range(self.n_prototypes):
            prototypes.append(
                self._find_classwise_prototype(matrix, prototypes)
            )
        return prototypes

    def select_prototypes(self, x: pd.DataFrame, y: pd.DataFrame) -> dict[int | str, pd.DataFrame]:
        classes = set()
        for col in y:
            classes.update(set(y[col].unique()))
        prototypes = {cls: [] for cls in classes}

        distances = self._create_distance_matrices(x, y, classes)

        for cls in classes:
            class_x = x[(y == cls).any(axis=1)]
            indices = self._find_single_class_prototypes(distances[cls])
            prototypes[cls] = class_x.iloc[indices]
        return prototypes


class SM_A(IExplainer):
    def __init__(self, model: _SKLearnRandomForestClassifier, n_prototypes: int = 3, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.n_prototypes = int(n_prototypes)

    def _find_classwise_prototype(self, matrix: np.ndarray, prototypes: list[int]) -> tuple[int, float]:
        if not prototypes:
            idx = np.argmin(matrix.sum(axis=1))
            improvement = matrix.shape[0] - np.sum(matrix[idx])
            return idx, improvement

        mask = np.in1d(range(matrix.shape[0]), prototypes)
        current_partial_distances = matrix[mask].min(axis=0)
        original_distance = current_partial_distances.sum()
        candidates = matrix[~mask]
        candidate_distances = np.minimum(candidates, current_partial_distances).sum(axis=1)
        idx = np.argmin(candidate_distances)
        improvement = original_distance - candidate_distances[idx]
        original_idx = np.where(~mask)[0][np.argmin(candidate_distances)]
        return original_idx, improvement

    def _find_prototype(self, distances: dict[int | str, np.ndarray], prototypes: dict[int | str, list[int]]
                        ) -> tuple[int | float, int]:
        prototype: tuple[float, int | str, int] = (-np.inf, -1, -1)
        for cls in distances:
            idx, improvement = self._find_classwise_prototype(distances[cls], prototypes[cls])
            if improvement > prototype[0]:
                prototype = (improvement, cls, idx)
        return prototype[1], prototype[2]

    def select_prototypes(self, x: pd.DataFrame, y: pd.DataFrame) -> dict[int | str, pd.DataFrame]:
        classes = set()
        for col in y:
            classes.update(set(y[col].unique()))
        prototypes = {cls: [] for cls in classes}

        distances = self._create_distance_matrices(x, y, classes)
        for _ in range(self.n_prototypes):
            cls, idx = self._find_prototype(distances, prototypes)
            prototypes[cls].append(idx)

        for cls in classes:
            prototypes[cls] = x[y.values == cls].iloc[prototypes[cls], :]

        return prototypes


class SM_WA(SM_A):
    def __init__(self, n_prototypes: int = 3, *args, **kwargs) -> None:
        super().__init__(n_prototypes=n_prototypes, *args, **kwargs)

    def _find_prototype(self, distances: dict[int | str, np.ndarray], prototypes: dict[int | str, list[int]]
                        ) -> tuple[int | float, int]:
        prototype: tuple[float, int | str, int] = (-np.inf, -1, -1)
        for cls in distances:
            idx, improvement = self._find_classwise_prototype(distances[cls], prototypes[cls])
            improvement = (1 / distances[cls].shape[0]) * improvement
            if improvement > prototype[0]:
                prototype = (improvement, cls, idx)
        return prototype[1], prototype[2]


class SG(SM_A):
    def __init__(self, n_prototypes: int = 3, *args, **kwargs) -> None:
        super().__init__(n_prototypes=n_prototypes, *args, **kwargs)

    def _find_prototype(self, prototypes: dict[int | str, list[int]], x: pd.DataFrame, y: pd.DataFrame, accuracy: float
                        ) -> tuple[int | str, int, float]:
        prototype: tuple[int | str, int] = (-1, -1)
        for cls in prototypes:
            class_x = x[y.values == cls]
            for idx in range(len(class_x)):
                if idx not in prototypes[cls]:
                    temp_prototypes = prototypes.copy()
                    temp_prototypes[cls].append(idx)
                    for _cls in temp_prototypes:
                        temp_prototypes[_cls] = x[y.values == _cls].iloc[temp_prototypes[_cls], :]
                    temp_accuracy = self.score_with_prototypes(x, y, temp_prototypes)
                    if temp_accuracy > accuracy:
                        prototype = (cls, idx)
        return prototype[1], prototype[2], accuracy

    def select_prototypes(self, x: pd.DataFrame, y: pd.DataFrame) -> dict[int | str, pd.DataFrame]:
        classes = set()
        for col in y:
            classes.update(set(y[col].unique()))
        prototypes = {cls: [] for cls in classes}
        accuracy = 0

        for _ in range(self.model.n_prototypes):
            cls, idx, accuracy = self._find_prototype(prototypes, x, y, accuracy)
            prototypes[cls].append(idx)

        for cls in classes:
            prototypes[cls] = x[y.values == cls].iloc[prototypes[cls], :]

        return prototypes


class APete(SM_A):
    def __init__(self, model: _SKLearnRandomForestClassifier, beta: float = 0.05) -> None:
        super().__init__(model)
        self.beta = float(beta)

    def _find_prototype(self, distances: dict[int | str, np.ndarray], prototypes: dict[int | str, list[int]]
                        ) -> tuple[int | float, int, float]:
        delta_prim = -np.inf
        prototype_idx = -1
        prototype_cls = -1
        for cls in distances:
            idx, delta = self._find_classwise_prototype(distances[cls], prototypes[cls])
            if delta > delta_prim:
                delta_prim = delta
                prototype_idx = idx
                prototype_cls = cls
        return prototype_cls, prototype_idx, delta_prim

    def select_prototypes(self, x: pd.DataFrame, y: pd.DataFrame) -> dict[int | str, pd.DataFrame]:
        prev_improvement = 0.0
        classes = set()
        for col in y:
            classes.update(set(y[col].unique()))
        prototypes = {cls: [] for cls in classes}

        distances = self._create_distance_matrices(x, y, classes)
        while True:
            cls, idx, improvement = self._find_prototype(distances, prototypes)
            prototypes[cls].append(idx)

            protos = {}
            for cls in classes:
                protos[cls] = x[y.values == cls].iloc[prototypes[cls], :]

            if np.abs(prev_improvement - improvement) / improvement <= self.beta:
                return protos
            prev_improvement = improvement
