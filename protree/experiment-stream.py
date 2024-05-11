from copy import deepcopy

import click
from river import forest
from river.preprocessing import StandardScaler

from protree import TPrototypes, TDataBatch
from protree.data import TDynamicDataset, DynamicDatasetFactory
from protree.explainers import TExplainer, Explainer
from protree.meta import RANDOM_SEED
from protree.metrics.compare import mutual_information, mean_minimal_distance, mean_centroid_displacement, \
    centroids_displacements, classwise_mean_minimal_distance
from protree.utils import pprint_dict


def create_comparison_dict(a: TPrototypes, b: TPrototypes, x: TDataBatch) -> dict[str, float | list[float]]:
    a, b = deepcopy(a), deepcopy(b)
    return {
        "total_n_prototypes (a)": sum(len(a[cls]) for cls in a),
        "classwise_n_prototypes (a)": {cls: len(a[cls]) for cls in a},
        "total_n_prototypes (b)": sum(len(b[cls]) for cls in b),
        "classwise_n_prototypes (b)": {cls: len(b[cls]) for cls in b},
        "mutual_information": mutual_information(a, b, x),
        "mean_minimal_distance": mean_minimal_distance(a, b),
        "classwise_mean_minimal_distance": classwise_mean_minimal_distance(a, b),
        "mean_centroid_displacement": mean_centroid_displacement(a, b),
        "centroids_displacements": centroids_displacements(a, b),
    }


@click.command()
@click.argument("dataset", type=click.Choice(TDynamicDataset.__args__))
@click.argument("explainer", type=click.Choice(["KMeans", "G_KM", "SM_A", "SM_WA", "SG", "APete"]))
@click.option("--n_trees", "-t", default=300, help="Number of trees. Allowable values are positive ints.")
@click.option("--kw_args", "-kw", type=str, default="",
              help="Additional, keyword arguments for the explainer. Must be in the form of key=value,key2=value2...")
@click.option("--memory_size", "-ms", type=int, default=800, help="The size of the memory.")
@click.option("--drift_position", "-dp", type=int, default=1500, help="The position of the drift.")
@click.option("--drift_width", "-dw", type=int, default=10, help="The width of the drift.")
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(dataset: TDynamicDataset, explainer, n_trees: int, kw_args: str, memory_size: int, drift_position: int,
         drift_width: int, log: bool) -> None:
    kw_args_dict = dict([arg.split("=") for arg in (kw_args.split(",") if kw_args else [])])
    ds = DynamicDatasetFactory.create(name=dataset, drift_position=drift_position, drift_width=drift_width)
    scaler = StandardScaler()
    model = forest.ARFClassifier(seed=RANDOM_SEED, n_models=n_trees, leaf_prediction="mc")

    pre_0 = [(scaler.learn_one(x) or scaler.transform_one(x), y) for (x, y) in ds.take(drift_position - int(memory_size))]
    pre_1 = [(scaler.learn_one(x) or scaler.transform_one(x), y) for (x, y) in ds.take(int(memory_size / 2))]
    pre_2 = [(scaler.learn_one(x) or scaler.transform_one(x), y) for (x, y) in ds.take(int(memory_size / 2))]
    post = [(scaler.learn_one(x) or scaler.transform_one(x), y) for (x, y) in ds.take(int(memory_size / 2))]

    # phase 1: model adaptation
    for x, y in pre_0 + pre_1:
        model.learn_one(x, y)

    # phase 2: prototype selection
    explainer_pre_1: TExplainer = Explainer[explainer].value(model=model, **kw_args_dict)
    prototypes_pre_1 = explainer_pre_1.select_prototypes([x for x, _ in pre_1])

    # phase 3: model adaptation
    for x, y in pre_2:
        model.learn_one(x, y)

    # phase 4: prototype selection
    explainer_pre_2: TExplainer = Explainer[explainer].value(model=model, **kw_args_dict)
    prototypes_pre_2 = explainer_pre_2.select_prototypes([x for x, _ in pre_2])

    pre_statistics = {"pre": create_comparison_dict(prototypes_pre_1, prototypes_pre_2, [x for x, _ in pre_2])}

    # phase 5: model adaptation after drift
    for x, y in post:
        model.learn_one(x, y)

    # phase 6: post drift prototype selection
    explainer_post: TExplainer = Explainer[explainer].value(model=model, **kw_args_dict)
    prototypes_post = explainer_post.select_prototypes([x for x, _ in post])

    post_statistics = {"post": create_comparison_dict(prototypes_pre_2, prototypes_post, [x for x, _ in post])}

    prototypes_dict = {
        "prototypes/pre_1": prototypes_pre_1,
        "prototypes/pre_2": prototypes_pre_2,
        "prototypes/post": prototypes_post
    }

    if log:
        pass

    pprint_dict(prototypes_dict)
    pprint_dict(pre_statistics)
    pprint_dict(post_statistics)


if __name__ == "__main__":
    main()
