import numpy as np
import pandas as pd

from protree.explainers.utils import _type_to_np_dtype


def get_euclidean_predictions(prototypes: dict[str | int, pd.DataFrame], x: pd.DataFrame) -> np.ndarray:
    predictions = (np.ones((len(x), 1)) * (-1)).astype(_type_to_np_dtype(prototypes))
    distance = np.ones((len(x))) * np.inf

    for cls in prototypes:
        for idx in prototypes[cls].index:
            dist = np.linalg.norm(x - prototypes[cls].loc[idx], axis=1)
            mask = dist < distance
            predictions[mask] = cls
            distance[mask] = dist[mask]
    predictions = predictions.flatten()
    return predictions


def mutual_information(a: dict[str | int, pd.DataFrame], b: dict[str | int, pd.DataFrame], x: pd.DataFrame) -> float:
    from sklearn.metrics import adjusted_mutual_info_score

    predictions_a = get_euclidean_predictions(a, x)
    predictions_b = get_euclidean_predictions(b, x)

    return adjusted_mutual_info_score(predictions_a, predictions_b, average_method="arithmetic")


def mean_minimal_distance(a: dict[str | int, pd.DataFrame], b: dict[str | int, pd.DataFrame],
                          penalty: float = 2.0) -> float:
    classes = set(a.keys()).union(b.keys())
    distances = []
    for cls in classes:
        # both empty
        if all([a[cls].shape[0] == 0, b[cls].shape[0] == 0]):
            continue

        # one empty
        elif a[cls].shape[0] == 0:
            distances.extend([penalty] * b[cls].shape[0])
        elif b[cls].shape[0] == 0:
            distances.extend([penalty] * a[cls].shape[0])

        # both not empty
        else:
            axis = int(a[cls].shape[0] > b[cls].shape[0])
            distance_matrix = np.linalg.norm(a[cls].values[:, None, ...] - b[cls].values[None, ...], axis=-1)
            distances.extend(distance_matrix.min(axis=axis).tolist())
    return np.mean(distances)
