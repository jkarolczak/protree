from copy import deepcopy

import numpy as np
import pandas as pd

from models.random_forest import PrototypicRandomForestClassifier


def individual_contribution(
        prototypes: dict[int | str, pd.DataFrame],
        cls: str | int,
        idx: int,
        model: PrototypicRandomForestClassifier,
        x: pd.DataFrame
) -> float:
    from protree.metrics.group import fidelity_with_model

    prototypes_without = deepcopy(prototypes)
    prototypes_without[cls].drop(idx, inplace=True)

    if not sum([len(c) for c in prototypes_without.values()]):
        raise ValueError("Cannot find the prototype of given index in the given class.")

    result = fidelity_with_model(prototypes, model, x) - fidelity_with_model(prototypes_without, model, x)
    return result


def voting_frequency(
        prototypes: dict[int | str, pd.DataFrame],
        cls: str | int,
        idx: int,
        model: PrototypicRandomForestClassifier,
        x: pd.DataFrame
) -> float:
    prototypes_flat = pd.concat([prototypes[c] for c in prototypes])
    re_idx = prototypes_flat.index.to_list().index(idx)

    x_leaves = model.apply(x)
    proto_leaves = model.apply(prototypes_flat)
    neighbourhood = []
    for i in range(len(proto_leaves)):
        neighbourhood.append((x_leaves == proto_leaves[i]).sum(axis=1))
    return (np.vstack(neighbourhood).argmax(axis=0) == re_idx).sum() / len(x)


def correct_votes(
        prototypes: dict[int | str, pd.DataFrame],
        cls: str | int,
        idx: int,
        model: PrototypicRandomForestClassifier,
        x: pd.DataFrame
) -> float:
    prototypes_flat = pd.concat([prototypes[c] for c in prototypes])
    re_idx = prototypes_flat.index.to_list().index(idx)

    classification = model.predict(x)
    mask = classification == cls

    x_leaves = model.apply(x)
    proto_leaves = model.apply(prototypes_flat)
    neighbourhood = []
    for i in range(len(proto_leaves)):
        neighbourhood.append((x_leaves == proto_leaves[i]).sum(axis=1))
    votes = (np.vstack(neighbourhood).argmax(axis=0) == re_idx)
    correct = np.logical_and(votes, mask).sum()
    return correct / votes.sum()


def hubness(
        prototypes: dict[int | str, pd.DataFrame],
        cls: str | int,
        idx: int,
        model: PrototypicRandomForestClassifier,
        x: pd.DataFrame,
        y: pd.DataFrame
) -> float:
    sub_x = x[y["target"] == cls]

    re_idx = prototypes[cls].index.to_list().index(idx)

    x_cls_leaves = model.apply(sub_x)
    proto_leaves = model.apply(prototypes[cls])

    neighbourhood = []
    for i in range(len(proto_leaves)):
        neighbourhood.append((x_cls_leaves == proto_leaves[i]).sum(axis=1))
    return (np.vstack(neighbourhood).argmax(axis=0) == re_idx).sum() / len(sub_x)


def _mean_similarity(
        sub_x: pd.DataFrame,
        prototypes: dict[int | str, pd.DataFrame],
        cls: str | int,
        idx: int,
        model: PrototypicRandomForestClassifier
) -> float:
    x_cls_leaves = model.apply(sub_x)
    proto_leaves = model.apply(pd.DataFrame(prototypes[cls].loc[idx]).transpose())
    return (x_cls_leaves == proto_leaves).sum() / np.prod(x_cls_leaves.shape)


def individual_in_distribution(
        prototypes: dict[int | str, pd.DataFrame],
        cls: str | int,
        idx: int,
        model: PrototypicRandomForestClassifier,
        x: pd.DataFrame,
        y: pd.DataFrame
) -> float:
    sub_x = x[y["target"] == cls]
    return _mean_similarity(sub_x, prototypes, cls, idx, model)


def individual_out_distribution(
        prototypes: dict[int | str, pd.DataFrame],
        cls: str | int,
        idx: int,
        model: PrototypicRandomForestClassifier,
        x: pd.DataFrame,
        y: pd.DataFrame
) -> float:
    sub_x = x[y["target"] != cls]
    return _mean_similarity(sub_x, prototypes, cls, idx, model)
