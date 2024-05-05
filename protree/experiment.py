from argparse import ArgumentParser, BooleanOptionalAction

import wandb

from protree.data import Dataset, DEFAULT_DATA_DIR
from protree.meta import N_JOBS
from protree.metrics.group import (fidelity_with_model, contribution, entropy_hubness, mean_in_distribution,
                                   mean_out_distribution)
from protree.models.random_forest import SmaTspRf
from protree.utils import parse_int_float_str, pprint_dict

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
        lazy=False,
        normalise=True
    )
    model = SmaTspRf(
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
                "max_features": max_features,
                "n_prototypes": args.n_prototypes,
            }
        )

    model.fit(*ds.train)
    prototypes = model.select_prototypes(*ds.train)

    statistics = {
        "n prototypes": sum([len(c) for c in prototypes.values()]),
        "score/valid/default": model.score(ds.valid[0], ds.valid[1]),
        "score/valid/prototypes": model.score_with_prototypes(ds.valid[0], ds.valid[1], prototypes),
        "score/test/default": model.score(ds.test[0], ds.test[1]),
        "score/test/prototypes": model.score_with_prototypes(ds.test[0], ds.test[1], prototypes),
        "score/train/fidelity (with model)": fidelity_with_model(prototypes, model, ds.train[0]),
        "score/train/contribution": contribution(prototypes, model, ds.train[0]),
        "score/train/hubness": entropy_hubness(prototypes, model, ds.train[0], ds.train[1]),
        "score/train/mean_in_distribution": mean_in_distribution(prototypes, model, ds.train[0], ds.train[1]),
        "score/train/mean_out_distribution": mean_out_distribution(prototypes, model, ds.train[0], ds.train[1])
    }

    if args.log:
        wandb.log(statistics)
        wandb.finish(quiet=True)
    else:
        pprint_dict(statistics)
