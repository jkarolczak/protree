from typing import Callable

import pandas as pd

from explainers.tree_distance import IExplainer
from metrics.classification import balanced_accuracy


def fidelity_with_model(
        prototypes: dict[int | str, pd.DataFrame],
        explainer: IExplainer,
        x: pd.DataFrame
) -> float:
    y_model = explainer.model.model.predict(x)
    y_prototypes = explainer.get_prototypes_predictions(x, prototypes)
    return balanced_accuracy(y_model, y_prototypes)


def contribution(
        prototypes: dict[int | str, pd.DataFrame],
        explainer: IExplainer,
        x: pd.DataFrame
) -> float:
    from protree.metrics.individual import individual_contribution

    diversity = 0
    n_prototypes = 0
    for cls in prototypes:
        for idx in prototypes[cls].index:
            try:
                diversity += individual_contribution(prototypes, cls, idx, explainer, x)
                n_prototypes += 1
            except ValueError:
                continue
    return diversity / n_prototypes


def _dist(
        prototypes: dict[int | str, pd.DataFrame],
        explainer: IExplainer,
        x: pd.DataFrame,
        y: pd.DataFrame,
        func: Callable
) -> float:
    dist = 0
    n_prototypes = 0
    for cls in prototypes:
        for idx in prototypes[cls].index:
            dist += func(prototypes, cls, idx, explainer, x, y)
            n_prototypes += 1
    return dist / n_prototypes


def mean_in_distribution(
        prototypes: dict[int | str, pd.DataFrame],
        explainer: IExplainer,
        x: pd.DataFrame,
        y: pd.DataFrame
) -> float:
    from protree.metrics.individual import individual_in_distribution
    return _dist(prototypes, explainer, x, y, individual_in_distribution)


def mean_out_distribution(
        prototypes: dict[int | str, pd.DataFrame],
        explainer: IExplainer,
        x: pd.DataFrame,
        y: pd.DataFrame
) -> float:
    from protree.metrics.individual import individual_out_distribution
    return _dist(prototypes, explainer, x, y, individual_out_distribution)


def entropy_hubness(
        prototypes: dict[int | str, pd.DataFrame],
        explainer: IExplainer,
        x: pd.DataFrame,
        y: pd.DataFrame
) -> float:
    from numpy import log
    from scipy.stats import entropy

    from protree.metrics.individual import hubness

    hub_score = {cls: [] for cls in prototypes}
    for cls in prototypes:
        for idx in prototypes[cls].index:
            hub_score[cls].append(hubness(prototypes, cls, idx, explainer, x, y))
    entropies = [entropy(hub_score[cls]) / max(log(len(hub_score[cls]) + 1e-6), 1 + 1e-6) for cls in hub_score]
    mean_entropy = sum(entropies) / len(entropies)
    return mean_entropy
