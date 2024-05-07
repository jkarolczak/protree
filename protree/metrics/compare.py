import numpy as np

from protree import TDataBatch, TPrototypes
from protree.explainers.utils import _type_to_np_dtype
from protree.utils import parse_batch, parse_prototypes


def get_euclidean_predictions(prototypes: TPrototypes, x: TDataBatch) -> np.ndarray:
    prototypes = parse_prototypes(prototypes)
    x = parse_batch(x)

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


def mutual_information(a: TPrototypes, b: TPrototypes, x: TDataBatch) -> float:
    from sklearn.metrics import adjusted_mutual_info_score

    predictions_a = get_euclidean_predictions(a, x)
    predictions_b = get_euclidean_predictions(b, x)

    return adjusted_mutual_info_score(predictions_a, predictions_b, average_method="arithmetic")


def mean_minimal_distance(a: TPrototypes, b: TPrototypes, penalty: float = 2.0) -> float:
    classes = set(a.keys()).union(b.keys())
    distances = []
    for cls in classes:
        # both empty
        if all([len(a[cls]) == 0, len(b[cls]) == 0]):
            continue

        # one empty
        elif len(a[cls]) == 0:
            distances.extend([penalty] * len(b[cls]))
        elif len(b[cls]) == 0:
            distances.extend([penalty] * len(a[cls]))

        # both not empty
        else:
            axis = int(len(a[cls]) > len(b[cls]))
            a[cls] = parse_batch(a[cls])
            b[cls] = parse_batch(b[cls])
            distance_matrix = np.linalg.norm(a[cls].values[:, None, ...] - b[cls].values[None, ...], axis=-1)
            distances.extend(distance_matrix.min(axis=axis).tolist())
    return np.mean(distances)
