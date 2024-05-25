import click
import matplotlib.pyplot as plt
import wandb
from matplotlib.colors import ListedColormap
from river import forest
from river.metrics import BalancedAccuracy

from protree.data.river import TDynamicDataset, DynamicDatasetFactory
from protree.data.stream_generators import TStreamGenerator, StreamGeneratorFactory
from protree.meta import RANDOM_SEED

FILE_NAME = "acc.png"

colors = ["cyan", "magenta"]
cmap = ListedColormap(colors)


@click.command()
@click.argument("dataset", type=click.Choice(TDynamicDataset.__args__ + TStreamGenerator.__args__))
@click.option("--n_trees", "-t", default=100, help="Number of trees. Allowable values are positive ints.")
@click.option("--drift_position", "-dp", type=int, default=[1150, 1800, 2500], multiple=True,
              help="The position of the drift.")
@click.option("--drift_width", "-dw", type=int, default=1, help="The width of the drift.")
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(dataset: TDynamicDataset | TStreamGenerator, n_trees: int, drift_position: list[int], drift_width: int,
         log: bool) -> None:
    if dataset in TStreamGenerator.__args__:
        ds = StreamGeneratorFactory.create(name=dataset, drift_position=drift_position, drift_width=drift_width)
    elif dataset in TDynamicDataset.__args__:
        ds = DynamicDatasetFactory.create(name=dataset, drift_position=drift_position, drift_width=drift_width)
    model = forest.ARFClassifier(seed=RANDOM_SEED, n_models=n_trees, leaf_prediction="mc")

    if log:
        wandb.init(
            project="Protree",
            entity="jacek-karolczak",
            name=f"arf-{dataset}-drift",
            config={
                "experiment-type": "stream-drift-arf",
                "dataset": dataset,
                "n_estimators": n_trees,
                "drift_position": drift_position,
                "drift_width": drift_width,
            }
        )

    drift_history = []
    acc_history = [0]

    acc = BalancedAccuracy()

    for i, (x, y) in enumerate(ds):
        y_pred = model.predict_one(x)
        acc.update(y, y_pred)
        acc_history.append(acc.get())

        model.learn_one(x, y)

        if i == max(drift_position) + 400:
            break

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
    fig.tight_layout()
    ax.vlines(x=drift_position, linewidth=2, color="black", label="Drift position", linestyle="dashed", alpha=0.5, ymin=0,
              ymax=max(acc_history) + 0.05)
    ax.plot(acc_history[1:], label="Balanced accuracy", c=colors[1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()

    plt.savefig(FILE_NAME, dpi=200)
    if log:
        wandb.log({
            "drifts": wandb.Image(FILE_NAME),
            "detected_drifts": drift_history
        })
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
