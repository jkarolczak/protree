from copy import deepcopy

import click
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from protree import TPrototypes, TDataBatch, TTarget
from protree.data.river import TDynamicDataset
from protree.data.stream_generators import TStreamGenerator, StreamGeneratorFactory
from protree.explainers import TExplainer, Explainer
from protree.explainers.tree_distance import IExplainer
from protree.meta import RANDOM_SEED
from protree.metrics.compare import mutual_information, mean_minimal_distance, mean_centroid_displacement, rand_index, \
    centroids_displacements, classwise_mean_minimal_distance, swap_delta, completeness, fowlkes_mallows
from protree.utils import pprint_dict


def create_comparison_dict(a: TPrototypes, b: TPrototypes, x: TDataBatch, y: TTarget, explainer_a: IExplainer | None = None,
                           explainer_b: IExplainer | None = None) -> dict[str, float | list[float]]:
    a, b = deepcopy(a), deepcopy(b)

    comparison_dict = {
        "total_n_prototypes (a)": sum(len(a[cls]) for cls in a),
        "classwise_n_prototypes (a)": {cls: len(a[cls]) for cls in a},
        "total_n_prototypes (b)": sum(len(b[cls]) for cls in b),
        "classwise_n_prototypes (b)": {cls: len(b[cls]) for cls in b},
        "mean_minimal_distance": mean_minimal_distance(a, b),
        "classwise_mean_minimal_distance": classwise_mean_minimal_distance(a, b),
        "mean_centroid_displacement": mean_centroid_displacement(a, b),
        "centroids_displacements": centroids_displacements(a, b),
        "swap_delta (l2)": swap_delta(a, b, x, y),
        "swap_delta (tree distance)": swap_delta(a, b, x, y, explainer=explainer_b),
    }

    for metric in [mutual_information, rand_index, completeness, fowlkes_mallows]:
        comparison_dict[f"{metric.__name__} (tree distance, prototypes)"] = metric(
            a, b, x, explainer_a=explainer_a, explainer_b=explainer_b, assign_to="prototype"
        )
        comparison_dict[f"{metric.__name__} (tree distance, classes)"] = metric(
            a, b, x, explainer_a=explainer_a, explainer_b=explainer_b, assign_to="class"
        )
        comparison_dict[f"{metric.__name__} (l2, prototypes)"] = metric(
            a, b, x, assign_to="prototype"
        )
        comparison_dict[f"{metric.__name__} (l2, classes)"] = metric(
            a, b, x, assign_to="class"
        )

    return comparison_dict


@click.command()
@click.argument("dataset", type=click.Choice(TStreamGenerator.__args__))
@click.argument("explainer", type=click.Choice(["KMeans", "G_KM", "SM_A", "SM_WA", "SG", "APete"]))
@click.option("--n_trees", "-t", default=300, help="Number of trees. Allowable values are positive ints.")
@click.option("--kw_args", "-kw", type=str, default="",
              help="Additional, keyword arguments for the explainer. Must be in the form of key=value,key2=value2...")
@click.option("--block_size", "-bs", type=int, default=2000, help="The size of the block.")
@click.option("--drift_width", "-dw", type=int, default=1, help="The width of the drift.")
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(dataset: TDynamicDataset | TStreamGenerator, explainer, n_trees: int, kw_args: str, block_size: int, drift_width: int,
         log: bool) -> None:
    kw_args_dict = dict([arg.split("=") for arg in (kw_args.split(",") if kw_args else [])])
    ds = StreamGeneratorFactory.create(name=dataset, drift_position=2 * block_size, drift_duration=drift_width)
    ds = list(ds.take(3 * block_size))
    x = pd.DataFrame.from_records([x for x, _ in ds])
    y = pd.DataFrame({"target": [y for _, y in ds]})

    scaler = MinMaxScaler()
    x[x.columns.tolist()] = scaler.fit_transform(x.values)

    pre_0 = (x.iloc[:block_size, :], y.iloc[:block_size, :])
    pre_1 = (x.iloc[block_size:2 * block_size, :], y.iloc[block_size:2 * block_size, :])
    post = (x.iloc[2 * block_size:3 * block_size, :], y.iloc[2 * block_size:3 * block_size, :])

    # phase 1: model adaptation
    model_pre_0 = RandomForestClassifier(n_estimators=n_trees, random_state=RANDOM_SEED).fit(*pre_1)

    # phase 2: prototype selection
    explainer_pre_0: TExplainer = Explainer[explainer].value(model=model_pre_0, **kw_args_dict)
    prototypes_pre_0 = explainer_pre_0.select_prototypes(pre_0[0])

    if log:
        wandb.init(
            project="Protree",
            entity="jacek-karolczak",
            name=f"{type(explainer_pre_0).__name__}-{dataset}-{kw_args}-{n_trees}",
            config={
                "experiment-type": "stream-sklearn",
                "explainer": type(explainer).__name__,
                "dataset": dataset,
                "n_estimators": n_trees,
                "block_size": block_size,
                **kw_args_dict
            }
        )

    # phase 3: model adaptation
    model_pre_1 = RandomForestClassifier(n_estimators=n_trees, random_state=RANDOM_SEED).fit(*pre_1)

    # phase 4: prototype selection
    explainer_pre_1: TExplainer = Explainer[explainer].value(model=model_pre_1, **kw_args_dict)
    prototypes_pre_1 = explainer_pre_1.select_prototypes(pre_1[0])

    pre_statistics = create_comparison_dict(prototypes_pre_0, prototypes_pre_1, *pre_1, explainer_pre_0, explainer_pre_1)

    # phase 5: model adaptation after drift
    model_post = RandomForestClassifier(n_estimators=n_trees, random_state=RANDOM_SEED).fit(*post)

    # phase 6: post drift prototype selection
    explainer_post: TExplainer = Explainer[explainer].value(model=model_post, **kw_args_dict)
    prototypes_post = explainer_post.select_prototypes(post[0])

    post_statistics = create_comparison_dict(prototypes_pre_1, prototypes_post, *post, explainer_pre_1, explainer_post)

    statistics = {
        "pre-pre": pre_statistics,
        "pre-post": post_statistics
    }

    if log:
        wandb.log(statistics)
        wandb.finish(quiet=True)
    else:
        pprint_dict(statistics)


if __name__ == "__main__":
    main()
