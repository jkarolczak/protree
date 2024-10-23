import click
import wandb
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier

from protree.data.static import StationaryDataset, DEFAULT_DATA_DIR, TStationaryDataset
from protree.explainers import Explainer, TExplainer
from protree.meta import N_JOBS, RANDOM_SEED
from protree.metrics.group import vector_voting_frequency, fidelity_with_model, mean_entropy_hubness, mean_in_distribution, \
    vector_entropy_hubness, vector_out_distribution, vector_consistent_votes, mean_out_distribution, vector_in_distribution
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
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(dataset: TStationaryDataset, explainer, directory: str, n_features: str | int, n_trees: int, log: bool, kw_args: str):
    max_features = parse_int_float_str(n_features)
    kw_args_dict = dict([arg.split("=") for arg in (kw_args.split(",") if kw_args else [])])
    ds = StationaryDataset(
        name=dataset,
        directory=directory,
        lazy=False,
        normalise=True
    )
    model = RandomForestClassifier(n_estimators=n_trees, max_features=max_features, n_jobs=N_JOBS, random_state=RANDOM_SEED)
    explainer: TExplainer = Explainer[explainer].value(model=model, **kw_args_dict)

    if log:
        wandb.init(
            project="Protree",
            entity="jacek-karolczak",
            name=f"{type(explainer).__name__}-{dataset}-{kw_args}-{n_trees}-{max_features}",
            config={
                "explainer": type(explainer).__name__,
                "dataset": dataset,
                "n_estimators": n_trees,
                "max_features": max_features,
                **kw_args_dict
            }
        )

    model.fit(ds.train[0], ds.train[1]["target"].ravel())
    prototypes = explainer.select_prototypes(ds.train[0])

    gmean_average = "micro"  # "micro" if dataset == "rhc" else "binary"

    statistics = {
        "total_n_prototypes": sum([len(c) for c in prototypes.values()]),
        "score/accuracy/train/random_forest": explainer.score(ds.train[0], ds.train[1]),
        "score/accuracy/train/prototypes": explainer.score_with_prototypes(ds.train[0], ds.train[1], prototypes),
        "score/accuracy/valid/random_forest": explainer.score(ds.valid[0], ds.valid[1]),
        "score/accuracy/valid/prototypes": explainer.score_with_prototypes(ds.valid[0], ds.valid[1], prototypes),
        "score/accuracy/test/random_forest": explainer.score(ds.test[0], ds.test[1]),
        "score/accuracy/test/prototypes": explainer.score_with_prototypes(ds.test[0], ds.test[1], prototypes),

        "score/gmean/train/random_forest": geometric_mean_score(ds.train[1], model.predict(ds.train[0]),
                                                                average=gmean_average),
        "score/gmean/train/prototypes": geometric_mean_score(ds.train[1],
                                                             explainer.predict_with_prototypes(ds.train[0], prototypes),
                                                             average=gmean_average),
        "score/gmean/valid/random_forest": geometric_mean_score(ds.valid[1], model.predict(ds.valid[0]),
                                                                average=gmean_average),
        "score/gmean/valid/prototypes": geometric_mean_score(ds.valid[1],
                                                             explainer.predict_with_prototypes(ds.valid[0], prototypes),
                                                             average=gmean_average),
        "score/gmean/test/random_forest": geometric_mean_score(ds.test[1], model.predict(ds.test[0]), average=gmean_average),
        "score/gmean/test/prototypes": geometric_mean_score(ds.test[1],
                                                            explainer.predict_with_prototypes(ds.test[0], prototypes),
                                                            average=gmean_average),

        "score/valid/fidelity": fidelity_with_model(prototypes, explainer, ds.valid[0]),
        "score/valid/hubness": mean_entropy_hubness(prototypes, explainer, ds.valid[0]),
        "score/valid/mean_in_distribution": mean_in_distribution(prototypes, explainer, *ds.valid),
        "score/valid/mean_out_distribution": mean_out_distribution(prototypes, explainer, *ds.valid),
        "vector/valid/partial_in_distribution": vector_in_distribution(prototypes, explainer, *ds.valid),
        "vector/valid/partial_hubnesses": vector_entropy_hubness(prototypes, explainer, ds.valid[0]),
        "vector/valid/partial_out_distribution": vector_out_distribution(prototypes, explainer, *ds.valid),
        "vector/valid/consistent_votes": vector_consistent_votes(prototypes, explainer, ds.valid[0]),
        "vector/valid/voting_frequency": vector_voting_frequency(prototypes, explainer, ds.valid[0])
    }
    if log:
        wandb.log(statistics)
        wandb.finish(quiet=True)
    else:
        pprint_dict(statistics)


if __name__ == "__main__":
    main()
