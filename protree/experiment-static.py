import click
import wandb
from sklearn.ensemble import RandomForestClassifier

from protree.data import Dataset, DEFAULT_DATA_DIR, TDataset
from protree.explainers import Explainer, TExplainer
from protree.meta import N_JOBS, RANDOM_SEED
from protree.metrics.group import (fidelity_with_model, contribution, entropy_hubness, mean_in_distribution,
                                   mean_out_distribution)
from protree.utils import parse_int_float_str, pprint_dict


@click.command()
@click.argument("dataset", type=click.Choice(["breast_cancer", "caltech", "compass", "diabetes", "mnist", "rhc"]))
@click.argument("explainer", type=click.Choice(["KMeans", "G_KM", "SM_A", "SM_WA", "SG", "APete"]))
@click.option("--directory", "-d", default=DEFAULT_DATA_DIR, help="Directory where datasets are stored.")
@click.option("--n_features", "-p", default="0.33", help="The number of features to consider when looking for the "
                                                         "best split. Allowable values are 'sqrt', positive ints and "
                                                         "floats between 0 and 1.")
@click.option("--n_trees", "-t", default=1000, help="Number of trees. Allowable values are positive ints.")
@click.option("--kw_args", "-kw", type=str, default="",
              help="Additional, keyword arguments for the explainer. Must be in the form of key=value,key2=value2...")
@click.option("--log", is_flag=True, help="Number of trees. Allowable values are positive ints.")
def main(dataset: TDataset, explainer, directory: str, n_features: str | int, n_trees: int, log: bool, kw_args: str):
    max_features = parse_int_float_str(n_features)
    kw_args_dict = dict([arg.split("=") for arg in kw_args.split(",")])
    ds = Dataset(
        name=dataset,
        directory=directory,
        lazy=False,
        normalise=True
    )
    model = RandomForestClassifier(n_estimators=n_trees, max_features=max_features, n_jobs=N_JOBS, random_state=RANDOM_SEED)
    explainer_cls: TExplainer = Explainer[explainer].value(model=model, **kw_args_dict)
    explainer = explainer_cls

    if log:
        wandb.init(
            project="Protree",
            entity="jacek-karolczak",
            name=f"{type(model).__name__}-{dataset}-{kw_args}-{n_trees}-{max_features}",
            config={
                "model": type(model).__name__,
                "dataset": dataset,
                "n_estimators": n_trees,
                "max_features": max_features,
                **kw_args_dict
            }
        )

    model.fit(ds.train[0], ds.train[1]["target"].ravel())
    prototypes = explainer.select_prototypes(*ds.train)

    statistics = {
        "n prototypes": sum([len(c) for c in prototypes.values()]),
        "score/valid/default": explainer.score(ds.valid[0], ds.valid[1]),
        "score/valid/prototypes": explainer.score_with_prototypes(ds.valid[0], ds.valid[1], prototypes),
        "score/test/default": explainer.score(ds.test[0], ds.test[1]),
        "score/test/prototypes": explainer.score_with_prototypes(ds.test[0], ds.test[1], prototypes),
        "score/train/fidelity (with model)": fidelity_with_model(prototypes, explainer, ds.train[0]),
        "score/train/contribution": contribution(prototypes, explainer, ds.train[0]),
        "score/train/hubness": entropy_hubness(prototypes, explainer, ds.train[0], ds.train[1]),
        "score/train/mean_in_distribution": mean_in_distribution(prototypes, explainer, ds.train[0], ds.train[1]),
        "score/train/mean_out_distribution": mean_out_distribution(prototypes, explainer, ds.train[0], ds.train[1])
    }

    if log:
        wandb.log(statistics)
        wandb.finish(quiet=True)
    else:
        pprint_dict(statistics)


if __name__ == "__main__":
    main()
