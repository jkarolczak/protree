import os

import click


def run_static_explanation() -> None:
    command_base = "python scripts/experiment-static.py --log -t 1000 -d ./data "

    for dataset, p, alg, kw in [
        ("breast_cancer", 0.33, "SM_A", "n_prototypes=18"),
        ("breast_cancer", 0.33, "SM_WA", "n_prototypes=15"),
        ("breast_cancer", 0.33, "KMeans", "n_prototypes=4"),
        ("breast_cancer", 0.33, "G_KM", "n_prototypes=4"),
        ("breast_cancer", 0.33, "APete", "alpha=0.05"),
        ("diabetes", "sqrt", "SM_A", "n_prototypes=4"),
        ("diabetes", "sqrt", "SM_WA", "n_prototypes=4"),
        ("diabetes", "sqrt", "G_KM", "n_prototypes=3"),
        ("diabetes", "sqrt", "KMeans", "n_prototypes=3"),
        ("diabetes", "sqrt", "APete", "alpha=0.05"),
        ("compass", "sqrt", "SM_A", "n_prototypes=13"),
        ("compass", "sqrt", "SM_WA", "n_prototypes=11"),
        ("compass", "sqrt", "KMeans", "n_prototypes=5"),
        ("compass", "sqrt", "G_KM", "n_prototypes=5"),
        ("compass", "sqrt", "APete", "alpha=0.01"),
        ("rhc", 0.7, "SM_A", "n_prototypes=13"),
        ("rhc", 0.7, "SM_WA", "n_prototypes=10"),
        ("rhc", 0.7, "G_KM", "n_prototypes=3"),
        ("rhc", 0.7, "KMeans", "n_prototypes=3"),
        ("rhc", 0.7, "APete", "alpha=0.01"),
        ("mnist", 0.33, "SM_A", "n_prototypes=12"),
        ("mnist", 0.33, "SM_WA", "n_prototypes=12"),
        ("mnist", 0.33, "G_KM", "n_prototypes=7"),
        ("mnist", 0.33, "KMeans", "n_prototypes=7"),
        ("mnist", 0.33, "APete", "alpha=0.05"),
        ("caltech", 0.7, "SM_A", "n_prototypes=6"),
        ("caltech", 0.7, "SM_WA", "n_prototypes=3"),
        ("caltech", 0.7, "G_KM", "n_prototypes=3"),
        ("caltech", 0.7, "KMeans", "n_prototypes=3"),
        ("caltech", 0.7, "APete", "alpha=0.05"),
    ]:
        command = command_base + f" -p {p} -kw {kw} {dataset} {alg}"
        print(command)
        os.system(command)


def run_drift_explanation_sklearn() -> None:
    command_base = "python scripts/experiment-stream-sklearn.py --log -t 300 -dw 1 -n 20"
    for dataset in ["sine", "random_tree", "rbf", "plane"]:
        for alg, kw in [
            *[("APete", f"alpha={n}") for n in (0.05,)],
        ]:
            for bs in [100, 1000, 10000, 25000]:
                command = command_base + f" -bs {bs} -kw {kw} {dataset} {alg}"
                print(command)
                os.system(command)


def run_drift_detection() -> None:
    command_base = "python scripts/experiment-detect-drift.py --log -t 200 -bs 2000 --kw_args=\"n_prototypes=5\" "
    for dataset in ["sine1", "sine500", "plane100", "plane1000", "random_tree20", "random_tree500", "rbf1", "sea1", "stagger1",
                    "mixed1"]:
        command = command_base + f"{dataset} G_KM"
        print(command)
        os.system(command)


@click.command()
@click.argument("experiment",
                type=click.Choice(["static-explanation", "drift-explanation-sklearn", "drift-detection"]))
def main(experiment: str) -> None:
    match experiment:
        case "static-explanation":
            run_static_explanation()
        case "drift-explanation-sklearn":
            run_drift_explanation_sklearn()
        case "drift-explanation":
            pass
        case "drift-detection":
            run_drift_detection()
        case _:
            raise ValueError("Invalid experiment type.")


if __name__ == "__main__":
    main()
