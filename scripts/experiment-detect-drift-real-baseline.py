import gc

import click
import wandb
from river import drift
from river import forest
from river import metrics

from protree.data.real_stream import TRealStream, RealStreamGeneratorFactory
from protree.meta import RANDOM_SEED


@click.command()
@click.argument("dataset", type=click.Choice(TRealStream.__args__))
@click.argument("detector", type=click.Choice(["PageHinkley", "ADWIN"]))
@click.option("--n_trees", "-t", default=100, help="Number of trees. Allowable values are positive ints.")
@click.option("--block_size", "-bs", type=int, default=1000, help="The size of the block.")
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(dataset: str, detector: str, n_trees: int, block_size: int, log: bool) -> None:
    model = forest.ARFClassifier(
        seed=RANDOM_SEED,
        n_models=n_trees,
        leaf_prediction="nba",
        grace_period=20,
        delta=0.1
    )

    if detector == "PageHinkley":
        detector_kwargs = {"delta": 0.005, "alpha": 0.001, "threshold": 50}
        drift_detector = drift.PageHinkley(**detector_kwargs)
    else:
        detector_kwargs = {"clock": 250, "grace_period": 40_000, "delta": 0.05, "min_window_length": 50}
        drift_detector = drift.ADWIN(**detector_kwargs)

    ds = RealStreamGeneratorFactory.create(name=dataset)
    accuracy_metric = metrics.Accuracy()

    if log:
        wandb.init(
            project="Protree",
            entity="jacek-karolczak",
            name=f"{detector}-{dataset}",
            config={
                "experiment-type": "real-stream",
                "detector": detector,
                "dataset": dataset,
                "n_estimators": n_trees,
                "block_size": block_size,
                **detector_kwargs
            }
        )

    for idx in range(len(ds)):
        x, y = ds.take(1)[0]
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        accuracy_metric.update(y, y_pred)
        accuracy = accuracy_metric.get()
        drift_detector.update(accuracy)

        if log:
            wandb.log({
                "metric": accuracy,
                "drift_detected": drift_detector.drift_detected
            })

        gc.collect()

    if log:
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
