from typing import Callable

import pandas as pd

from protree import TPrototypes, TDataBatch, TTarget
from protree.explainers.tree_distance import IExplainer
from protree.metrics.classification import balanced_accuracy


def fidelity_with_model(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch) -> float:
    y_model = explainer.model.get_model_predictions(x)
    y_prototypes = explainer.get_prototypes_predictions(x, prototypes)
    return balanced_accuracy(y_model, y_prototypes)


def contribution(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch) -> float:
    from protree.metrics.individual import individual_contribution

    diversity = 0
    n_prototypes = 0
    for cls in prototypes:
        for idx in (prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls]))):
            try:
                diversity += individual_contribution(prototypes, cls, idx, explainer, x)
                n_prototypes += 1
            except ValueError:
                continue
    return diversity / n_prototypes


def _dist(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch, y: TTarget, func: Callable) -> float:
    dist = 0
    n_prototypes = 0
    for cls in prototypes:
        for idx in (prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls]))):
            dist += func(prototypes, cls, idx, explainer, x, y)
            n_prototypes += 1
    return dist / n_prototypes


def mean_in_distribution(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch, y: TTarget) -> float:
    from protree.metrics.individual import individual_in_distribution
    return _dist(prototypes, explainer, x, y, individual_in_distribution)


def mean_out_distribution(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch, y: TTarget) -> float:
    from protree.metrics.individual import individual_out_distribution
    return _dist(prototypes, explainer, x, y, individual_out_distribution)


def entropy_hubness(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch, y: TTarget) -> float:
    from numpy import log
    from scipy.stats import entropy

    from protree.metrics.individual import hubness

    hub_score = {cls: [] for cls in prototypes}
    for cls in prototypes:
        for idx in (prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls]))):
            hub_score[cls].append(hubness(prototypes, cls, idx, explainer, x, y))
    entropies = [entropy(hub_score[cls]) / max(log(len(hub_score[cls]) + 1e-6), 1 + 1e-6) for cls in hub_score]
    mean_entropy = sum(entropies) / len(entropies)
    return mean_entropy
