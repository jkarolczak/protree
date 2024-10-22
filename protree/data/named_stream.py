from typing import TypeAlias, Literal

from protree.data.river_generators import Agrawal, SEA, RBF
from protree.data.stream_generators import Sine, Plane, RandomTree

TNamedStream: TypeAlias = Literal[
    "sine1", "sine500", "plane100", "plane1000", "random_tree20", "random_tree500", "RBF1", "RBF4000", "agrawal1",
    "agrawal200", "SEA1", "SEA1000"]


class INamedStream:
    pass


class NamedStreamGeneratorFactory:
    @staticmethod
    def create(name: TNamedStream) -> INamedStream:
        name = "".join([n.capitalize() for n in name.split("_")])
        return globals()[f"{name}"]()


class Sine1(Sine, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[42900, 55500, 67200, 81100], drift_duration=1, seed=42, informative_attrs=(3, 2))


class Sine500(Sine, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[41300, 60000, 73000], drift_duration=500, seed=42, informative_attrs=(3, 2))


class Plane100(Plane, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[55500, 77200, 83500, 94000], drift_duration=100, seed=42)


class Plane1000(Plane, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[48500, 72200, 88700], drift_duration=1000, seed=42)


class RandomTree20(RandomTree, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[42100, 49800, 79500], drift_duration=20, seed=42, n_informative=4, n_redundant=2)


class RandomTree500(RandomTree, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[50000, 70000, 90000], drift_duration=500, seed=42, n_informative=2, n_redundant=1)


class Rbf1(RBF, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[47100, 63000, 88800], drift_duration=1, seed=42, n_informative=4, n_centroids=9)


class Rbf4000(RBF, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[56000, 76000], drift_duration=4000, seed=42, n_informative=4, n_centroids=11)


class Agrawal1(Agrawal, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[50000, 72200, 89000], drift_duration=1, classification_function=0, seed=42)


class Agrawal200(Agrawal, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[43000, 62000, 81200], drift_duration=200, seed=42, classification_function=2)


class Sea1(SEA, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[43900, 65000, 77500, 90000], drift_duration=1, seed=42)


class Sea1000(SEA, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[45200, 79500], drift_duration=1000, seed=42)
