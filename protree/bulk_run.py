import os

import click


def run_static_explanation() -> None:
    command_base = "python protree/experiment-static.py --log -t 1000 -d ./data "


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
    command_base = "python protree/experiment-stream-sklearn.py --log -t 300 -cs 2000 -dw 1 "
    for dataset in ["sine", "random_tree", "make_classification", "plane"]:
        for alg, kw in [
            *[("APete", f"alpha={n}") for n in (0.05, 0.01)],
        ]:
            command = command_base + f" -kw {kw} {dataset} {alg}"
            print(command)
            os.system(command)


def run_drift_detection() -> None:
    command_base = "python protree/experiment-detect-drift.py --log -t 500 -dp 450 -dp 800 -dp 1200 -dp 1800 -dw 1 "
    for dataset in ["sine", "random_tree", "make_classification", "plane"]:
        for ws in [200, 300, 400]:
            for m, a in (
                    *[("minimal_distance", a) for a in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]],
                    *[("centroid_displacement", a) for a in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]],
                    *[("mutual_info", a) for a in [0.01, 0.05, 0.1, 0.15, 0.2]],
            ):
                for kw in [f"alpha={n}" for n in (0.05, 0.01)]:
                    command = command_base + f" -ws {ws} -a {a} -kw {kw} {dataset} APete"
                    print(command)
                    os.system(command)


@click.command()
@click.argument("experiment",
                type=click.Choice(["static-explanation", "drift-explanation-sklearn", "drift-explanation", "drift-detection"]))
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
