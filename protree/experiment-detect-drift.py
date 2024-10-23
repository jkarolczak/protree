import gc
from typing import Literal

import click
import wandb
from river import forest

from protree.data.named_stream import TNamedStream, NamedStreamGeneratorFactory
from protree.detectors import ProtoBDrift
from protree.explainers import Explainer
from protree.meta import RANDOM_SEED


@click.command()
@click.argument("dataset", type=click.Choice(TNamedStream.__args__))
@click.argument("explainer", type=click.Choice(["KMeans", "G_KM", "SM_A", "SM_WA", "SG", "APete"]))
@click.option("--n_trees", "-t", default=100, help="Number of trees. Allowable values are positive ints.")
@click.option("--kw_args", "-kw", type=str, default="",
              help="Additional, keyword arguments for the explainer. Must be in the form of key=value,key2=value2...")
@click.option("--block_size", "-bs", type=int, default=2000, help="The size of the block.")
@click.option("--measure", "-m", type=click.Choice(["mutual_information", "rand_index", "completeness",
                                                    "fowlkes_mallows", "centroid_displacement", "minimal_distance",
                                                    "swap_delta"]),
              default="swap_delta", help="The measure to use for prototype selection.")
@click.option("--strategy", "-s", type=click.Choice(["class", "total"]), default="total",
              help="The strategy to use for prototype selection.")
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(dataset: TNamedStream, explainer, n_trees: int, kw_args: str, block_size: int,
         measure: Literal["mutual_information", "centroid_displacement", "minimal_distance"],
         strategy: Literal["class", "total"], log: bool) -> None:
    kw_args_dict = dict([arg.split("=") for arg in (kw_args.split(",") if kw_args else [])])
    ds = NamedStreamGeneratorFactory.create(name=dataset)
    model = forest.ARFClassifier(seed=RANDOM_SEED, n_models=n_trees, leaf_prediction="nba", grace_period=20, delta=0.1)
    detector = ProtoBDrift(model=model, prototype_selector=Explainer[explainer].value, prototype_selector_kwargs=kw_args_dict,
                           measure=measure, strategy=strategy)

    if log:
        wandb.init(
            project="Protree",
            entity="jacek-karolczak",
            name=f"{explainer}-{dataset}-{kw_args}-drift-{measure}",
            config={
                "experiment-type": "stream-detect-drift",
                "explainer": explainer,
                "dataset": dataset,
                "n_estimators": n_trees,
                "drift_position": ds.drift_position,
                "drift_width": ds.drift_duration,
                "block_size": block_size,
                "measure": measure,
                **kw_args_dict
            }
        )

    drift_predictions = []

    i = 0

    while i * block_size < 100000:
        x_block, y_block = zip(*ds.take(block_size))
        for x, y in zip(x_block, y_block):
            model.learn_one(x, y)
        detector.update(x_block, y_block)
        if detector.drift_detected:
            drift_predictions.append(i * block_size)
            print(f"({i * block_size}) Drift detected!")

        i += 1
        gc.collect()

    false_alarms = sum(
        [not any([(d_hat - 1.5 * block_size) <= dp <= (d_hat + 1.5 * block_size) for dp in ds.drift_position]) for d_hat in
         drift_predictions])

    missed_drifts = sum(
        [not any([(d_hat - 1.5 * block_size) <= dp <= (d_hat + 1.5 * block_size) for d_hat in drift_predictions]) for dp in
         ds.drift_position])

    correctly_detected_drifts = sum(
        [any([(d_hat - 1.5 * block_size) <= dp <= (d_hat + 1.5 * block_size) for d_hat in drift_predictions]) for dp in
         ds.drift_position])

    print("Correctly detected drifts:", correctly_detected_drifts)
    print("False alarms:", false_alarms)
    print("Missed drifts:", missed_drifts)

    if log:
        wandb.log({
            "correctly_detected_drifts": correctly_detected_drifts,
            "false_alarms": false_alarms,
            "missed_drifts": missed_drifts,
            "detected_drifts": drift_predictions,
            "drift_positions": ds.drift_position
        })
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
