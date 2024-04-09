import warnings
from typing import Callable, Literal, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as _SKLearnRandomForestClassifier

from meta import RANDOM_SEED
from models.utils import parse_input, _type_to_np_dtype

C = 1000


class PrototypicRandomForestClassifier(_SKLearnRandomForestClassifier):
    def __init__(
            self,
            n_estimators: int = 100,
            criterion: Literal["gini", "entropy", "log_loss"] = "gini",
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int | float = 1,
            min_weight_fraction_leaf: int | float = 0.0,
            max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
            max_leaf_nodes: int | None = None,
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            oob_score: Callable[[np.ndarray, np.ndarray], float] | bool = False,
            n_jobs: int | None = None,
            random_state: int | None = RANDOM_SEED,
            verbose: int = 0,
            warm_start: bool = False,
            class_weight: Literal["balanced", "balanced_subsample"] | dict | list[dict] | None = None,
            ccp_alpha: float = 0.0,
            max_samples: int | float | None = None
    ) -> None:
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, class_weight=class_weight,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_samples=max_samples,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                         oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)

    def par_similarity(self, x1: np.ndarray | pd.Series, x2: np.ndarray | pd.Series) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            nodes1 = self.apply(parse_input(x1))
            nodes2 = self.apply(parse_input(x2))

        return sum((nodes1 == nodes2)[0]) / self.n_estimators

    def similarity_matrix(self, df: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nodes = self.apply(df)

        shape = df.shape[0]
        matrix = np.zeros((shape, shape), dtype=float)
        for i in range(shape):
            matrix[i] = (nodes[i] == nodes).sum(axis=1)
        matrix /= self.n_estimators
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

    def predict_with_prototypes(self, x: pd.DataFrame, prototypes: dict[int | str, pd.DataFrame]) -> np.ndarray:
        predictions = (np.ones((len(x), 1)) * (-1)).astype(_type_to_np_dtype(prototypes))
        similarity = np.zeros((len(x)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_nodes = self.apply(x)

        for cls in prototypes:
            for _, prototype in prototypes[cls].iterrows():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prototype_nodes = self.apply([prototype])
                prototype_similarity = (prototype_nodes == x_nodes).sum(axis=1) / self.n_estimators
                mask = prototype_similarity > similarity
                predictions[mask] = cls
                similarity[mask] = prototype_similarity[mask]
        return predictions

    @staticmethod
    def _score(y: pd.DataFrame, y_hat: np.ndarray) -> float:
        y = y.values.squeeze(1)
        acc = 0
        for label in np.unique(y):
            mask = (y == label)
            acc_partial = (y[mask] == y_hat[mask]).mean()
            acc += (acc_partial * mask.sum()) / len(y)
        return acc

    def score_with_prototypes(self, x: pd.DataFrame, y: pd.DataFrame, prototypes: dict[int | str, pd.DataFrame]) -> float:
        y_hat = self.predict_with_prototypes(x, prototypes)
        return PrototypicRandomForestClassifier._score(y, y_hat)

    def score(self, x: pd.DataFrame, y: pd.DataFrame, sample_weight: pd.DataFrame | None = None) -> float:
        y_hat = self.predict(x)
        return PrototypicRandomForestClassifier._score(y, y_hat)


class KMedoidTspRf(PrototypicRandomForestClassifier):
    def __init__(
            self,
            n_prototypes: int = 3,
            n_estimators: int = 100,
            criterion: Literal["gini", "entropy", "log_loss"] = "gini",
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int | float = 1,
            min_weight_fraction_leaf: int | float = 0.0,
            max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
            max_leaf_nodes: int | None = None,
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            oob_score: Callable[[np.ndarray, np.ndarray], float] | bool = False,
            n_jobs: int | None = None,
            random_state: int | None = RANDOM_SEED,
            verbose: int = 0,
            warm_start: bool = False,
            class_weight: Literal["balanced", "balanced_subsample"] | dict | list[dict] | None = None,
            ccp_alpha: float = 0.0,
            max_samples: int | float | None = None
    ) -> None:
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, class_weight=class_weight,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_samples=max_samples,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                         oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)
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


class SmaTspRf(PrototypicRandomForestClassifier):
    def __init__(
            self,
            n_prototypes: int = 3,
            n_estimators: int = 100,
            criterion: Literal["gini", "entropy", "log_loss"] = "gini",
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int | float = 1,
            min_weight_fraction_leaf: int | float = 0.0,
            max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
            max_leaf_nodes: int | None = None,
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            oob_score: Callable[[np.ndarray, np.ndarray], float] | bool = False,
            n_jobs: int | None = None,
            random_state: int | None = RANDOM_SEED,
            verbose: int = 0,
            warm_start: bool = False,
            class_weight: Literal["balanced", "balanced_subsample"] | dict | list[dict] | None = None,
            ccp_alpha: float = 0.0,
            max_samples: int | float | None = None
    ) -> None:
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, class_weight=class_weight,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_samples=max_samples,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                         oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)
        self.n_prototypes = n_prototypes

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


class SmwaTspRf(SmaTspRf):
    def __init__(
            self,
            n_prototypes: int = 3,
            n_estimators: int = 100,
            criterion: Literal["gini", "entropy", "log_loss"] = "gini",
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int | float = 1,
            min_weight_fraction_leaf: int | float = 0.0,
            max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
            max_leaf_nodes: int | None = None,
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            oob_score: Callable[[np.ndarray, np.ndarray], float] | bool = False,
            n_jobs: int | None = None,
            random_state: int | None = RANDOM_SEED,
            verbose: int = 0,
            warm_start: bool = False,
            class_weight: Literal["balanced", "balanced_subsample"] | dict | list[dict] | None = None,
            ccp_alpha: float = 0.0,
            max_samples: int | float | None = None
    ) -> None:
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, class_weight=class_weight,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_samples=max_samples,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                         oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                         n_prototypes=n_prototypes)

    def _find_prototype(self, distances: dict[int | str, np.ndarray], prototypes: dict[int | str, list[int]]
                        ) -> tuple[int | float, int]:
        prototype: tuple[float, int | str, int] = (-np.inf, -1, -1)
        for cls in distances:
            idx, improvement = self._find_classwise_prototype(distances[cls], prototypes[cls])
            improvement = (1 / distances[cls].shape[0]) * improvement
            if improvement > prototype[0]:
                prototype = (improvement, cls, idx)
        return prototype[1], prototype[2]


class SgTspRf(SmaTspRf):
    def __init__(
            self,
            n_prototypes: int = 3,
            n_estimators: int = 100,
            criterion: Literal["gini", "entropy", "log_loss"] = "gini",
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int | float = 1,
            min_weight_fraction_leaf: int | float = 0.0,
            max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
            max_leaf_nodes: int | None = None,
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            oob_score: Callable[[np.ndarray, np.ndarray], float] | bool = False,
            n_jobs: int | None = None,
            random_state: int | None = RANDOM_SEED,
            verbose: int = 0,
            warm_start: bool = False,
            class_weight: Literal["balanced", "balanced_subsample"] | dict | list[dict] | None = None,
            ccp_alpha: float = 0.0,
            max_samples: int | float | None = None
    ) -> None:
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, class_weight=class_weight,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_samples=max_samples,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                         oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                         n_prototypes=n_prototypes)

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

        for _ in range(self.n_prototypes):
            cls, idx, accuracy = self._find_prototype(prototypes, x, y, accuracy)
            prototypes[cls].append(idx)

        for cls in classes:
            prototypes[cls] = x[y.values == cls].iloc[prototypes[cls], :]

        return prototypes


class APete(SmaTspRf):
    def __init__(
            self,
            beta: float = 0.05,
            n_estimators: int = 100,
            criterion: Literal["gini", "entropy", "log_loss"] = "gini",
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int | float = 1,
            min_weight_fraction_leaf: int | float = 0.0,
            max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
            max_leaf_nodes: int | None = None,
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            oob_score: Callable[[np.ndarray, np.ndarray], float] | bool = False,
            n_jobs: int | None = None,
            random_state: int | None = RANDOM_SEED,
            verbose: int = 0,
            warm_start: bool = False,
            class_weight: Literal["balanced", "balanced_subsample"] | dict | list[dict] | None = None,
            ccp_alpha: float = 0.0,
            max_samples: int | float | None = None
    ) -> None:
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, class_weight=class_weight,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_samples=max_samples,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                         oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                         n_prototypes=-1)
        self.beta = beta

    def _find_prototype(self, distances: dict[int | str, np.ndarray], prototypes: dict[int | str, list[int]]
                        ) -> tuple[int | float, int, float]:
        prototype: tuple[float, int | str, int] = (-np.inf, -1, -1)
        for cls in distances:
            idx, improvement = self._find_classwise_prototype(distances[cls], prototypes[cls])
            if improvement > prototype[0]:
                prototype = (improvement, cls, idx)
        return prototype[1], prototype[2], prototype[0]

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
