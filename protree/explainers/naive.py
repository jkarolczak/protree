from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import protree.explainers.utils
from protree.explainers.tree_distance import IExplainer
from protree.explainers.utils import _type_to_np_dtype
from protree.meta import RANDOM_SEED


class KMeans:
    def __init__(
            self,
            n_prototypes: int = 3,
            init: str = "k-means++",
            n_init: str = "auto",
            max_iter: int = 300,
            tol: float = 0.0001,
            verbose: int = 0,
            random_state: int | None = RANDOM_SEED,
            copy_x: bool = True,
            algorithm: str = "lloyd",
            *args,
            **kwargs
    ) -> None:
        self.prototypes = {}
        self._kmeans = {}
        self._kwargs = dict(n_clusters=n_prototypes, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose,
                            random_state=random_state, copy_x=copy_x, algorithm=algorithm)

    def _fit_select_prototypes(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        classes = y[y.columns[0]].unique()
        for c in classes:
            is_in_c = (y[y.columns[0]] == c).to_list()
            x_partial = x[is_in_c]

            c_kwargs = self._kwargs.copy()
            c_kwargs["n_clusters"] = min(c_kwargs["n_clusters"], len(x_partial))

            self._kmeans[c] = KMeans(**c_kwargs).fit(x_partial)
            cluster_centers = self._kmeans[c].cluster_centers_
            distances = np.linalg.norm(x_partial.values[:, np.newaxis, :] - cluster_centers[np.newaxis, :], axis=2)
            closest_indices = np.argmin(distances, axis=0)
            self.prototypes[c] = protree.explainers.utils.iloc[closest_indices]

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> KMeans:
        if not self.prototypes:
            self._fit_select_prototypes(x, y)
        return self

    def select_prototypes(self, x: pd.DataFrame, y: pd.DataFrame) -> dict[int | str, pd.DataFrame]:
        if not self.prototypes:
            self._fit_select_prototypes(x, y)
        return self.prototypes

    def predict_with_prototypes(self, x: pd.DataFrame, prototypes: dict[int | str, pd.DataFrame]) -> np.ndarray:
        predictions = (np.ones((len(x), 1)) * (-1)).astype(_type_to_np_dtype(prototypes))
        distance = np.ones((len(x))) * np.inf
        for cls in prototypes:
            for _, cluster_centers in prototypes[cls].iterrows():
                distances = np.linalg.norm(x.values[:, np.newaxis, :] - cluster_centers.values[np.newaxis, :],
                                           axis=2).squeeze()
                mask = distances < distance
                predictions[mask] = cls
                distance = np.minimum(distances, distance)
        return predictions

    def score_with_prototypes(self, x: pd.DataFrame, y: pd.DataFrame, prototypes: dict[int | str, pd.DataFrame]) -> float:
        y_hat = self.predict_with_prototypes(x, prototypes)
        return IExplainer._score(y, y_hat)
