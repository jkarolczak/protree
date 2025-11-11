import gc
import json
from typing import Literal

import click
import wandb
from river import forest

from protree.data.real_stream import TRealStream, RealStreamGeneratorFactory
from protree.detectors import RaceP
from protree.explainers import Explainer
from protree.meta import RANDOM_SEED


@click.command()
@click.argument("dataset", type=click.Choice(TRealStream.__args__))
@click.argument("explainer", type=click.Choice(["KMeans", "G_KM", "SM_A", "SM_WA", "SG", "APete"]))
@click.option("--n_trees", "-t", default=100, help="Number of trees. Allowable values are positive ints.")
@click.option("--kw_args", "-kw", type=str, default="",
              help="Additional, keyword arguments for the explainer. Must be in the form of key=value,key2=value2...")
@click.option("--block_size", "-bs", type=int, default=1000, help="The size of the block.")
@click.option("--measure", "-m", type=click.Choice(["mutual_information", "rand_index", "completeness",
                                                    "fowlkes_mallows", "centroid_displacement", "minimal_distance",
                                                    "prototype_reassignment_impact"]),
              default="prototype_reassignment_impact", help="The measure to use for prototype selection.")
@click.option("--strategy", "-s", type=click.Choice(["class", "total"]), default="total",
              help="The strategy to use for prototype selection.")
@click.option("--distance", "-d", type=click.Choice(["tree", "l2"]), default="tree",
              help="Distance to use we.")
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(dataset: str, explainer, n_trees: int, kw_args: str, block_size: int, distance: Literal["tree", "l2"],
         measure: Literal["mutual_information", "centroid_displacement", "minimal_distance"],
         strategy: Literal["class", "total"], log: bool) -> None:
    kw_args_dict = dict([arg.split("=") for arg in (kw_args.split(",") if kw_args else [])])
    model = forest.ARFClassifier(seed=RANDOM_SEED, n_models=n_trees, leaf_prediction="nba", grace_period=20, delta=0.1)
    detector = RaceP(model=model, prototype_selector=Explainer[explainer].value, prototype_selector_kwargs=kw_args_dict,
                     measure=measure, strategy=strategy)
    ds = RealStreamGeneratorFactory.create(name=dataset)

    if log:
        wandb.init(
            project="Protree",
            entity="jacek-karolczak",
            name=f"{explainer}-{dataset}-{kw_args}-drift-{measure}",
            config={
                "experiment-type": "real-stream",
                "explainer": explainer,
                "dataset": dataset,
                "n_estimators": n_trees,
                "block_size": block_size,
                "measure": measure,
                **kw_args_dict
            }
        )

    drift_predictions = []

    i = 0

    while i * block_size < len(ds) - block_size:
        x_block, y_block = zip(*ds.take(block_size))
        for x, y in zip(x_block, y_block):
            model.learn_one(x, y)
        detector.update(x_block, y_block)

        if log:
            wandb.log({
                "metric": detector.metric_value,
                "drift_detected": detector.drift_detected
            }, step=i * block_size)

        if detector.drift_detected:
            drift_predictions.append(i * block_size)
            print(f"({i * block_size}) Drift detected!")
            if log:
                wandb.log({
                    "prototypes_before": json.dumps(detector.prototypes[0]),
                    "prototypes_after": json.dumps(detector.prototypes[1]),
                })

        i += 1
        gc.collect()

    if log:
        wandb.log({
            "reference_mean": detector._reference_mean,
            "reference_std": detector._reference_std
        })
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
