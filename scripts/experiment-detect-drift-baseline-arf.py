import gc

import arff
import click
import pandas as pd
from river import metrics
from river.forest import ARFClassifier

from protree.data.named_stream import TNamedStream, NamedStreamGeneratorFactory
from protree.meta import RANDOM_SEED


@click.command()
@click.argument("dataset", type=click.Choice(TNamedStream.__args__))
@click.option("--n_trees", "-t", default=200, help="Number of trees. Allowable values are positive ints.")
def main(dataset: TNamedStream, n_trees: int) -> None:
    ds = NamedStreamGeneratorFactory.create(name=dataset)
    model = ARFClassifier(seed=RANDOM_SEED, n_models=n_trees, leaf_prediction="nba", grace_period=20, delta=0.1)
    accuracy_metric = metrics.Accuracy()
    csv_data = []

    for stream_position in range(100000):
        if stream_position % 1000 == 0:
            gc.collect()
            print(f"Stream position: {stream_position}")
        x, y = ds.take(1)[0]
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        accuracy_metric.update(y, y_pred)
        accuracy = accuracy_metric.get()

        row = {
            **x,
            "y": y,
            "stream_position": stream_position,
            "accuracy": accuracy
        }

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    path = f"data_arff/{dataset}.arff"
    arff.dump(path, df.values, relation=dataset, names=df.columns)


if __name__ == "__main__":
    main()
