import numpy as np
import pandas as pd

from protree import TDataBatch, TPrototypes
from protree.explainers.tree_distance import IExplainer
from protree.explainers.utils import _type_to_np_dtype
from protree.utils import parse_batch, parse_prototypes


def get_euclidean_predictions(prototypes: TPrototypes, x: TDataBatch) -> np.ndarray:
    """Generate predictions based on Euclidean distance between prototypes and the input data.

    :param prototypes: Prototypes.
    :type prototypes: TPrototypes
    :param x: Input data.
    :type x: TDataBatch

    :returns: Assignment of the input data to classes based on Euclidean distance to the nearest prototype.
    :rtype: np.ndarray

    """

    prototypes = parse_prototypes(prototypes)
    x = parse_batch(x)

    predictions = (np.ones((len(x), 1)) * (-1)).astype(_type_to_np_dtype(prototypes))
    distance = np.ones((len(x))) * np.inf

    for cls in prototypes:
        if isinstance(prototypes[cls], list):
            prototypes[cls] = pd.DataFrame.from_records(prototypes[cls])
        for idx in prototypes[cls].index:
            dist = np.linalg.norm(x - prototypes[cls].loc[idx], axis=1)
            mask = dist < distance
            predictions[mask] = cls
            distance[mask] = dist[mask]
    predictions = predictions.flatten()
    return predictions


def _get_predictions(a: TPrototypes, b: TPrototypes, x: TDataBatch, explainer: IExplainer | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    if explainer:
        predictions_a = explainer.predict_with_prototypes(x, a)
        predictions_b = explainer.predict_with_prototypes(x, b)
    else:
        predictions_a = get_euclidean_predictions(a, x)
        predictions_b = get_euclidean_predictions(b, x)

    return predictions_a, predictions_b


def mutual_information(a: TPrototypes, b: TPrototypes, x: TDataBatch, explainer: IExplainer | None = None) -> float:
    """Calculate the similarity metric based on mutual information between two sets of prototypes, with respect to the input
    data. If an explainer is provided, the predictions are based on the explainer, otherwise, the predictions are based on
    Euclidean distance.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param explainer: Explainer to generate predictions (Optional).
    :type explainer: IExplainer

    :returns: The similarity metric between the two sets of prototypes based on mutual information.
    :rtype: float

    """

    from sklearn.metrics import adjusted_mutual_info_score

    predictions_a, predictions_b = _get_predictions(a, b, x, explainer)

    return adjusted_mutual_info_score(predictions_a, predictions_b, average_method="arithmetic")


def rand_index(a: TPrototypes, b: TPrototypes, x: TDataBatch, explainer: IExplainer | None = None) -> float:
    """Calculate the similarity metric based on the Rand index between two sets of prototypes, with respect to the input
    data. If an explainer is provided, the predictions are based on the explainer, otherwise, the predictions are based on
    Euclidean distance.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param explainer: Explainer to generate predictions (Optional).
    :type explainer: IExplainer

    :returns: The similarity metric between the two sets of prototypes based on the Rand index.
    :rtype: float

    """

    from sklearn.metrics import adjusted_rand_score

    predictions_a, predictions_b = _get_predictions(a, b, x, explainer)

    return adjusted_rand_score(predictions_a, predictions_b)


def fowlkes_mallows(a: TPrototypes, b: TPrototypes, x: TDataBatch, explainer: IExplainer | None = None) -> float:
    """Calculate the similarity metric based on the Fowlkes-Mallows index between two sets of prototypes, with respect to the
    input data. If an explainer is provided, the predictions are based on the explainer, otherwise, the predictions are based
    on Euclidean distance.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param explainer: Explainer to generate predictions (Optional).
    :type explainer: IExplainer

    :returns: The similarity metric between the two sets of prototypes based on the Fowlkes-Mallows index.
    :rtype: float

    """

    from sklearn.metrics import fowlkes_mallows_score

    predictions_a, predictions_b = _get_predictions(a, b, x, explainer)

    return fowlkes_mallows_score(predictions_a, predictions_b)


def completeness(a: TPrototypes, b: TPrototypes, x: TDataBatch, explainer: IExplainer | None = None) -> float:
    """Calculate the similarity metric based on the completeness score between two sets of prototypes, with respect to the
    input data. If an explainer is provided, the predictions are based on the explainer, otherwise, the predictions are based
    on Euclidean distance.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param explainer: Explainer to generate predictions (Optional).
    :type explainer: IExplainer

    :returns: The similarity metric between the two sets of prototypes based on the completeness score.
    :rtype: float

    """

    from sklearn.metrics import completeness_score

    predictions_a, predictions_b = _get_predictions(a, b, x, explainer)

    return completeness_score(predictions_a, predictions_b)


def centroids_displacements(a: TPrototypes, b: TPrototypes, penalty: float = 10.0) -> dict[int | str, float]:
    """Calculate the displacement of the centroids between two sets of prototypes.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param penalty: Penalty for empty classes (default = 10.0).
    :type penalty: float
    """

    classes = set(a.keys()).union(b.keys())

    a = parse_prototypes(a)
    b = parse_prototypes(b)

    distances = {}

    for cls in classes:
        # both empty
        if (cls not in a or len(a[cls]) == 0) and (cls not in b or len(b[cls]) == 0):
            distances[cls] = 0.0

        # one empty
        elif len(a[cls]) == 0 or len(b[cls]) == 0:
            distances[cls] = penalty

        # both not empty
        else:
            distances[cls] = np.linalg.norm((a[cls].mean() - b[cls].mean()).values)
    return distances


def mean_centroid_displacement(a: TPrototypes, b: TPrototypes, penalty: float = 2.0) -> float:
    """Calculate the mean displacement of the centroids between two sets of prototypes.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param penalty: Penalty for empty classes (default = 2.0).
    :type penalty: float

    :returns: The mean displacement of the centroids between the two sets of prototypes.
    :rtype: float

    """

    displacements = centroids_displacements(a, b, penalty)
    return np.mean(list(displacements.values()))


def _l2_distance_matrix(a: TPrototypes, b: TPrototypes, cls: int | float) -> np.ndarray:
    axis = int(len(a[cls]) > len(b[cls]))
    a[cls] = parse_batch(a[cls])
    b[cls] = parse_batch(b[cls])
    distance_matrix = np.linalg.norm(a[cls].values[:, None, ...] - b[cls].values[None, ...], axis=-1)
    return distance_matrix.min(axis=axis).tolist()


def mean_minimal_distance(a: TPrototypes, b: TPrototypes, penalty: float = 2.0) -> float:
    """Calculate the mean minimal distance between two sets of prototypes.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param penalty: Penalty for empty classes (default = 2.0).
    :type penalty: float

    :returns: The mean minimal distance between the two sets of prototypes.
    :rtype: float

    """

    classes = set(a.keys()).union(b.keys())
    distances = []
    for cls in classes:
        # both empty
        if (cls not in a or len(a[cls]) == 0) and (cls not in b or len(b[cls]) == 0):
            continue

        # one empty
        elif cls not in a or len(a[cls]) == 0:
            distances.extend([penalty] * len(b[cls]))
        elif cls not in b or len(b[cls]) == 0:
            distances.extend([penalty] * len(a[cls]))

        # both not empty
        else:
            distances.extend(_l2_distance_matrix(a, b, cls))

    return np.mean(distances)


def classwise_mean_minimal_distance(a: TPrototypes, b: TPrototypes, penalty: float = 2.0) -> dict[int | str, float]:
    """Calculate the class-wise mean minimal distance between two sets of prototypes.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param penalty: Penalty for empty classes (default = 2.0).
    :type penalty: float

    :returns: The class-wise mean minimal distance between the two sets of prototypes.
    :rtype: dict[int | str, float]

    """
    
    classes = set(a.keys()).union(b.keys())
    distances = {}
    for cls in classes:
        # both empty
        if (cls not in a or len(a[cls]) == 0) and (cls not in b or len(b[cls]) == 0):
            distances[cls] = 0.0

        # one empty
        elif (cls not in b or len(a[cls]) == 0) or (cls not in b or len(b[cls]) == 0):
            distances[cls] = penalty * len(b[cls] if cls in b and b[cls] else a[cls])

        # both not empty
        else:
            distances[cls] = np.mean(_l2_distance_matrix(a, b, cls))
    return distances
