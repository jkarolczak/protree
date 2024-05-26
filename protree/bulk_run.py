import os

import click


def run_static_explanation() -> None:
    command_base = "python protree/experiment-static.py --log -t 1000 -d ./data "
    for dataset, p in [("breast_cancer", 0.33), ("diabetes", "sqrt"), ("compass", "sqrt"), ("rhc", 0.7), ("mnist", 0.33),
                       ("caltech", 0.7)]:
        for alg, kw in [
            *[("APete", f"beta={n}") for n in (0.05, 0.01)],
            *[("SM_A", f"n_prototypes={n}") for n in range(2, 21)],
            *[("SM_WA", f"n_prototypes={n}") for n in range(2, 21)],
            *[("G_KM", f"n_prototypes={n}") for n in range(1, 11)],
            *[("KMeans", f"n_prototypes={n}") for n in range(1, 11)],
        ]:
            command = command_base + f" -p {p} -kw {kw} {dataset} {alg}"
            print(command)
            os.system(command)


def run_drift_explanation_sklearn() -> None:
    command_base = "python protree/experiment-stream-sklearn.py --log -t 300 -cs 2000 -dw 1 "
    for dataset in ["sine", "random_tree", "make_classification", "plane"]:
        for alg, kw in [
            *[("APete", f"beta={n}") for n in (0.05, 0.01)],
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
                for kw in [f"beta={n}" for n in (0.05, 0.01)]:
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
