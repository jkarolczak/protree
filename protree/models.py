import warnings
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as _SKLearnRandomForestClassifier

from meta import RANDOM_SEED


def _1d_np_to_2d_np(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x, axis=0)


def parse_input(x: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    match type(x):
        case np.ndarray:
            match len(x.shape):
                case 1:
                    return _1d_np_to_2d_np(x)
                case 2:
                    return x
                case _:
                    raise ValueError("Invalid input size.")
        case pd.Series:
            return _1d_np_to_2d_np(x.to_numpy())
        case pd.DataFrame:
            return x.to_numpy()
        case _:
            raise TypeError("Invalid input type.")


class RandomForestClassifier(_SKLearnRandomForestClassifier):
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
        oob_score: Callable[[np.ndarray, np.ndarray], float] | bool= False,
        n_jobs: int | None = None,
        random_state: int | None = RANDOM_SEED,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Literal["balanced", "balanced_subsample"] | dict | list[dict] | None = None,
        ccp_alpha: float = 0.0,
        max_samples: int | float | None = None
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )
        self.n_prototypes = n_prototypes

    def par_similarity(self, x1: np.ndarray | pd.Series, x2: np.ndarray | pd.Series) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            nodes1 = self.apply(parse_input(x1))
            nodes2 = self.apply(parse_input(x2))

        return sum((nodes1 == nodes2)[0]) / self.n_estimators

    def similarity_matrix(self, df: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            nodes = self.apply(df)

        shape = df.shape[0]
        matrix = np.zeros((shape, shape), dtype=float)
        for i in range(shape):
            matrix[i] = (nodes[i] == nodes).sum(axis=1)
        matrix /= self.n_estimators
        return matrix

    def distance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return 1 - self.similarity_matrix(df)

    def _single_class_prototypes(self, df: pd.DataFrame) -> list[int]:
        matrix = self.distance_matrix(df)
        shape = matrix.shape[0]
        indices = []
        for i in range(min(self.n_prototypes, shape)):
            idx = np.argmin(
                np.sum(matrix, axis=1)
            )
            indices.append(idx)
            matrix[idx] = 10 * shape
        return indices

    def select_prototypes(self, x: pd.DataFrame, y: pd.DataFrame) -> dict[int | str, pd.DataFrame]:
        prototypes = {}
        classes = set()
        for col in y:
            classes.update(set(y[col].unique()))
        for cls in classes:
            class_x = x[(y == cls).any(axis=1)]
            indices = self._single_class_prototypes(class_x)
            prototypes[cls] = class_x.iloc[indices]
        return prototypes
