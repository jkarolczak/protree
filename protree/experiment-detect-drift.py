from typing import Literal

import click
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.colors import ListedColormap
from river import forest

from protree.data.river import TDynamicDataset, DynamicDatasetFactory
from protree.data.stream_generators import TStreamGenerator, StreamGeneratorFactory
from protree.detectors import Ancient
from protree.explainers import Explainer
from protree.meta import RANDOM_SEED

PLT_FILE_NAME = "drift.png"
colors = ["cyan", "magenta"]
cmap = ListedColormap(colors)


@click.command()
@click.argument("dataset", type=click.Choice(TDynamicDataset.__args__ + TStreamGenerator.__args__))
@click.argument("explainer", type=click.Choice(["KMeans", "G_KM", "SM_A", "SM_WA", "SG", "APete"]))
@click.option("--n_trees", "-t", default=100, help="Number of trees. Allowable values are positive ints.")
@click.option("--kw_args", "-kw", type=str, default="",
              help="Additional, keyword arguments for the explainer. Must be in the form of key=value,key2=value2...")
@click.option("--window_size", "-ws", type=int, default=400, help="The size of the window.")
@click.option("--alpha", "-a", type=float, default=0.5, help="The alpha parameter for drift detection.")
@click.option("--measure", "-m", type=click.Choice(["mutual_info", "centroid_displacement", "minimal_distance"]),
              default="centroid_displacement", help="The measure to use for prototype selection.")
@click.option("--strategy", "-s", type=click.Choice(["class", "total"]), default="class",
              help="The strategy to use for prototype selection.")
@click.option("--drift_position", "-dp", type=int, default=[1150, 1800, 2500], multiple=True,
              help="The position of the drift.")
@click.option("--drift_width", "-dw", type=int, default=1, help="The width of the drift.")
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(dataset: TDynamicDataset | TStreamGenerator, explainer, n_trees: int, kw_args: str, window_size: int,
         measure: Literal["mutual_info", "centroid_displacement", "minimal_distance"], strategy: Literal["class", "total"],
         alpha: float, drift_position: list[int], drift_width: int, log: bool) -> None:
    kw_args_dict = dict([arg.split("=") for arg in (kw_args.split(",") if kw_args else [])])
    if dataset in TStreamGenerator.__args__:
        ds = StreamGeneratorFactory.create(name=dataset, drift_position=drift_position, drift_width=drift_width)
    elif dataset in TDynamicDataset.__args__:
        ds = DynamicDatasetFactory.create(name=dataset, drift_position=drift_position, drift_width=drift_width)
    model = forest.ARFClassifier(seed=RANDOM_SEED, n_models=n_trees, leaf_prediction="mc")
    detector = Ancient(model=model, prototype_selector=Explainer[explainer].value, prototype_selector_kwargs=kw_args_dict,
                       window_length=window_size, alpha=alpha, measure=measure, strategy=strategy, clock=16)

    if log:
        wandb.init(
            project="Protree",
            entity="jacek-karolczak",
            name=f"{explainer}-{dataset}-{kw_args}-drift-{measure}-{alpha}",
            config={
                "experiment-type": "stream-detect-drift",
                "explainer": explainer,
                "dataset": dataset,
                "n_estimators": n_trees,
                "drift_position": drift_position,
                "drift_width": drift_width,
                "window_size": window_size,
                "measure": measure,
                "alpha": alpha,
                **kw_args_dict
            }
        )

    x_history = []
    y_history = []
    drift_history = []

    for i, (x, y) in enumerate(ds):
        x_history.append(x)
        y_history.append(y)
        model.learn_one(x, y)
        detector.update(x, y)
        if detector.drift_detected:
            drift_history.append(i)
            print(f"{int(i - window_size / 2)}) Drift detected!")

        if i == max(drift_position) + 400:
            break

    x_split = [[list(x_.values())[i] for x_ in x_history] for i in range(len(x_history[0]))]

    fig, ax = plt.subplots(nrows=int(np.ceil(len(x_split) / 2)), ncols=2, figsize=(11, 4.5))
    fig.tight_layout()
    for i, x_ in enumerate(x_split):
        ax[i // 2, i % 2].set_title(f"x{i}")
        ax[i // 2, i % 2].scatter(list(range(len(x_))), x_, c=y_history, label="Feature", cmap=cmap, s=2)
        if drift_width == 1:
            ax[i // 2, i % 2].vlines(x=drift_position, linewidth=2, color="black", linestyle="dashed", alpha=0.75,
                                     label="Real drift position", ymin=0, ymax=1)
            color = "black"
        else:
            for pos in drift_position:
                ax[i // 2, i % 2].fill_betweenx(y=[0, 1], x1=pos, x2=pos + drift_width, color="black", alpha=0.5,
                                                label="Real drift position")
            color = "orange"
        ax[i // 2, i % 2].vlines(x=[int(d - window_size / 2) for d in drift_history], linewidth=2, color=color, alpha=0.75,
                                 linestyle="solid", label="Detected drift", ymin=0, ymax=1)
        ax[i // 2, i % 2].spines["top"].set_visible(False)
        ax[i // 2, i % 2].spines["right"].set_visible(False)

    if len(x_split) % 2:
        ax[len(ax) - 1, 1].spines["top"].set_visible(False)
        ax[len(ax) - 1, 1].spines["right"].set_visible(False)
        ax[len(ax) - 1, 1].spines["bottom"].set_visible(False)
        ax[len(ax) - 1, 1].spines["left"].set_visible(False)
        ax[len(ax) - 1, 1].get_xaxis().set_ticks([])
        ax[len(ax) - 1, 1].get_yaxis().set_ticks([])

    false_alarms = sum([not any([d - window_size <= dp <= d for dp in drift_position]) for d in drift_history])
    missed_drifts = sum([not any([d - window_size <= dp <= d for d in drift_history]) for dp in drift_position])

    print("False alarms:", false_alarms)
    print("Missed drifts:", missed_drifts)

    plt.savefig(PLT_FILE_NAME, dpi=200)
    if log:
        wandb.log({
            "drifts": wandb.Image(PLT_FILE_NAME),
            "detected_drifts": drift_history
        })
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
