from argparse import ArgumentParser, BooleanOptionalAction

import wandb

from metrics.group import contribution, entropy_hubness, fidelity_with_model, mean_in_distribution, \
    mean_out_distribution
from metrics.individual import voting_frequency, correct_votes
from models.random_forest import SmaTspRf
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

    score_valid_with_prototypes = model.score_with_prototypes(ds.valid[0], ds.valid[1], prototypes)
    score_valid = model.score(ds.valid[0], ds.valid[1])

    score_test_with_prototypes = model.score_with_prototypes(ds.test[0], ds.test[1], prototypes)
    score_test = model.score(ds.test[0], ds.test[1])

    score_fidelity_with_model = fidelity_with_model(prototypes, model, ds.train[0])
    score_contribution = contribution(prototypes, model, ds.train[0])
    score_in_distribution = mean_in_distribution(prototypes, model, ds.train[0], ds.train[1])
    score_out_distribution = mean_out_distribution(prototypes, model, ds.train[0], ds.train[1])
    score_hubness = entropy_hubness(prototypes, model, ds.train[0], ds.train[1])

    print(voting_frequency(prototypes, 0, 8, model, ds.train[0]))
    print(correct_votes(prototypes, 0, 8, model, ds.train[0]))
    print(voting_frequency(prototypes, 0, 312, model, ds.train[0]))
    print(correct_votes(prototypes, 0, 312, model, ds.train[0]))
    print(voting_frequency(prototypes, 1, 35, model, ds.train[0]))
    print(correct_votes(prototypes, 1, 35, model, ds.train[0]))

    n_prototypes = sum([len(c) for c in prototypes.values()])

    if args.log:
        wandb.log(
            {
                "n prototypes": n_prototypes,
                "score/valid/default": score_valid,
                "score/valid/prototypes": score_valid_with_prototypes,
                "score/test/default": score_test,
                "score/test/prototypes": score_test_with_prototypes,
                "score/train/fidelity (with model)": score_fidelity_with_model,
                "score/train/contribution": score_contribution,
                "score/train/hubness": score_hubness,
                "score/train/mean_in_distribution": score_in_distribution,
                "score/train/mean_out_distribution": score_out_distribution
            }
        )

        wandb.finish(quiet=True)
    else:
        print(f"n prototypes: {n_prototypes}")
        print(f"score/valid/default: {score_valid:2.4f}")
        print(f"score/valid/prototypes: {score_valid_with_prototypes:2.4f}")
        print(f"score/test/default: {score_test:2.4f}")
        print(f"score/test/prototypes: {score_test_with_prototypes:2.4f}")
        print(f"fidelity (with model): {score_fidelity_with_model:2.4f}")
        print(f"contribution: {score_contribution:2.4f}")
        print(f"hubness: {score_hubness:2.4f}")
        print(f"mean in distribution: {score_in_distribution:2.4f}")
        print(f"mean out distribution: {score_out_distribution:2.4f}")
