from __future__ import annotations

from typing import TypeAlias, Literal

from river.datasets import synth

from protree.meta import RANDOM_SEED

TDynamicDataset: TypeAlias = Literal["mixed_dirft", "sea_drift", "sine_drift", "tree_drift", "agrawal_drift"]


class DynamicDatasetFactory:
    @staticmethod
    def create(name: TDynamicDataset, drift_position: int = 500, drift_width: int = 5, seed: int = RANDOM_SEED
               ) -> synth.ConceptDriftStream:
        return globals()[f"{''.join([n.capitalize() for n in name.split('-')])}"](drift_position=drift_position,
                                                                                  drift_width=drift_width, seed=seed)


class MixedDrift(synth.ConceptDriftStream):
    """Mixed synthetic dataset, based on the River library implementation:
    https://riverml.xyz/latest/api/datasets/synth/Mixed/

    The simulated drift is a transition between two classification functions available in the dataset.

    :param drift_position: The position of the drift.
    :param drift_width: The width of the drift.
    :param seed: Random seed.
    """

    def __init__(self, drift_position: int | tuple[int, ...] = 500, drift_width: int = 1, seed: int = RANDOM_SEED) -> None:
        super().__init__(stream=synth.Mixed(classification_function=0, balance_classes=False, seed=42),
                         drift_stream=synth.Mixed(classification_function=1, balance_classes=True, seed=42),
                         position=drift_position, width=drift_width, seed=seed)


class SeaDrift(synth.ConceptDriftStream):
    """SEA synthetic dataset, based on the River library implementation:
    https://riverml.xyz/latest/api/datasets/synth/SEA/

    The simulated drift is a transition between two variants of the original dataset.

    :param drift_position: The position of the drift.
    :param drift_width: The width of the drift.
    :param seed: Random seed.
    """

    def __init__(self, drift_position: int | tuple[int, ...] = 500, drift_width: int = 1, seed: int = RANDOM_SEED) -> None:
        super().__init__(stream=synth.SEA(variant=1, seed=42), drift_stream=synth.SEA(variant=2, seed=42),
                         position=drift_position, width=drift_width, seed=seed)


class SineDrift(synth.ConceptDriftStream):
    """Sine synthetic dataset, based on the River library implementation:
    https://riverml.xyz/latest/api/datasets/synth/Sine/

    The simulated drift is a transition between two classification functions available in the dataset.

    :param drift_position: The position of the drift.
    :param drift_width: The width of the drift.
    :param seed: Random seed.

    """

    def __init__(self, drift_position: int | tuple[int, ...] = 500, drift_width: int = 1, seed: int = RANDOM_SEED) -> None:
        super().__init__(stream=synth.Sine(classification_function=1, seed=42, balance_classes=True),
                         drift_stream=synth.Sine(classification_function=2, seed=42, balance_classes=True),
                         position=drift_position, width=drift_width, seed=seed)


class TreeDrift(synth.ConceptDriftStream):
    """RandomTree synthetic dataset, based on the River library implementation:
    https://riverml.xyz/latest/api/datasets/synth/RandomTree/

    The simulated drift is a transition between two trees.

    :param drift_position: The position of the drift.
    :param drift_width: The width of the drift.
    :param seed: Random seed.
    """

    def __init__(self, drift_position: int | tuple[int, ...] = 500, drift_width: int = 1, seed: int = RANDOM_SEED) -> None:
        super().__init__(stream=synth.RandomTree(seed_tree=42), drift_stream=synth.RandomTree(seed_tree=23),
                         position=drift_position, width=drift_width, seed=seed)


class AgrawalDrift(synth.ConceptDriftStream):
    """Agrawal synthetic dataset, based on the River library implementation:
    https://riverml.xyz/latest/api/datasets/synth/Agrawal/

    The simulated drift is a transition between two classification functions available in the dataset.

    :param drift_position: The position of the drift.
    :param drift_width: The width of the drift.
    :param seed: Random seed.
    """

    def __init__(self, drift_position: int | tuple[int, ...] = 500, drift_width: int = 1, seed: int = RANDOM_SEED) -> None:
        super().__init__(stream=synth.Agrawal(classification_function=1, seed=42),
                         drift_stream=synth.Agrawal(classification_function=6, seed=42),
                         position=drift_position, width=drift_width, seed=seed)
