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
    command_base = "python protree/experiment-stream-sklearn.py --log -t 300 -cs 750 -dw 10 "
    for dataset in ["mixed", "sea", "sine", "tree", "agrawal"]:
        for alg, kw in [
            *[("APete", f"beta={n}") for n in (0.05, 0.01)],
        ]:
            command = command_base + f" -kw {kw} {dataset} {alg}"
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
            pass
        case _:
            raise ValueError("Invalid experiment type.")


if __name__ == "__main__":
    main()
