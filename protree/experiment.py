from argparse import ArgumentParser, BooleanOptionalAction

import wandb

from models.random_forest import SgTspRf
from protree.data import Dataset, DEFAULT_DATA_DIR
from protree.meta import N_JOBS


def parse_int_float_str(value) -> int | float | str:
    try:
        return int(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    return value


parser = ArgumentParser()
parser.add_argument("dataset", choices=["breast_cancer", "caltech", "compass", "diabetes", "mnist", "rhc"],
                    help="The name of the dataset to evaluate.")
parser.add_argument("--directory", "-d", default=DEFAULT_DATA_DIR, help="Directory where datasets are stored.")
parser.add_argument("--n_features", "-p", default="0.33", help="The number of features to consider when looking for the "
                                                               "best split. Allowable values are 'sqrt', positive ints and "
                                                               "floats between 0 and 1.")
parser.add_argument("--n_trees", "-t", type=int, default=1000, help="Number of trees. Allowable values are positive ints.")
parser.add_argument("--n_prototypes", "-k", type=int, default=3, help="Number of prototypes to find. Allowable values are "
                                                                      "positive ints.")
parser.add_argument("--log", action=BooleanOptionalAction, help="Number of trees. Allowable values are positive ints.")

if __name__ == "__main__":
    args = parser.parse_args()
    max_features = parse_int_float_str(args.n_features)
    ds = Dataset(
        name=args.dataset,
        directory=args.directory,
        lazy=False
    )
    model = SgTspRf(
        n_prototypes=args.n_prototypes,
        n_estimators=args.n_trees,
        max_features=max_features,
        n_jobs=N_JOBS
    )
    if args.log:
        wandb.init(
            project="Protree",
            entity="jacek-karolczak",
            name=f"{type(model).__name__}-{args.dataset}-{args.n_prototypes}-{args.n_trees}-{max_features}",
            config={
                "model": type(model).__name__,
                "dataset": args.dataset,
                "n_estimators": args.n_trees,
                "max_features": max_features
            }
        )

    model.fit(*ds.train)
    prototypes = model.select_prototypes(*ds.train)

    score_valid_with_prototypes = model.score_with_prototypes(ds.valid[0], ds.valid[1], prototypes)
    score_valid = model.score(ds.valid[0], ds.valid[1])

    score_test_with_prototypes = model.score_with_prototypes(ds.test[0], ds.test[1], prototypes)
    score_test = model.score(ds.test[0], ds.test[1])

    if args.log:
        wandb.log(
            {
                "score/valid/default": score_valid,
                "score/valid/prototypes": score_valid_with_prototypes,
                "score/test/default": score_test,
                "score/test/prototypes": score_test_with_prototypes,
            }
        )

        wandb.finish(quiet=True)
    else:
        print(f"score/valid/default: {score_valid:2.4f}")
        print(f"score/valid/prototypes: {score_valid_with_prototypes:2.4f}")
        print(f"score/test/default: {score_test:2.4f}")
        print(f"score/test/prototypes: {score_test_with_prototypes:2.4f}")
