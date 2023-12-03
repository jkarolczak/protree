from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from meta import RANDOM_SEED

DEFAULT_DATA_DIR = "./data"


def _save_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR
) -> None:
    path = Path(directory) / f"{dataset_name}.csv"
    df.reset_index(drop=True).to_csv(path)


def _split_dataframe_and_save(
    df: pd.DataFrame,
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    train_size: float = 0.6,
    valid_size: float = 0.2,
    test_size: float = 0.2
) -> None:
    if train_size + valid_size + test_size != 1.0:
        raise ValueError("Sum of train, valid, and test sizes have to sum up to 1.0")

    from sklearn.model_selection import train_test_split

    y_column = [col for col in df.columns if "target" in col][0]

    train, valid_test = train_test_split(
        df,
        stratify=df[[y_column]],
        train_size=train_size,
        random_state=RANDOM_SEED
    )
    valid, test = train_test_split(
        valid_test,
        stratify=valid_test[[y_column]],
        train_size=valid_size/(valid_size + test_size),
        random_state=RANDOM_SEED
    )

    _save_dataframe(df=train, dataset_name=f"{dataset_name}_train", directory=directory)
    _save_dataframe(df=valid, dataset_name=f"{dataset_name}_valid", directory=directory)
    _save_dataframe(df=test, dataset_name=f"{dataset_name}_test", directory=directory)


def _download_diabetes(
    directory: str = DEFAULT_DATA_DIR
) -> None:
    from io import StringIO
    from requests import get

    response = get(
        url="https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        params={
            "downloadformat": "csv"
        }
    )

    df = pd.read_csv(StringIO(response.content.decode("utf-8")))
    df = df.rename(columns={
        "Outcome": "target"
    })
    _split_dataframe_and_save(df=df, dataset_name="diabetes", directory=directory)


def _download_breast_cancer(
    directory: str = DEFAULT_DATA_DIR
) -> None:
    from sklearn.datasets import load_breast_cancer

    breast_cancer = load_breast_cancer(as_frame=True)
    _split_dataframe_and_save(df=breast_cancer.frame, dataset_name="breast_cancer", directory=directory)


def _download_right_heart_catheterization(
    directory: str = DEFAULT_DATA_DIR
) -> None:
    from io import StringIO
    from requests import get

    response = get(
        url="https://hbiostat.org/data/repo/rhc.csv",
        params={
            "downloadformat": "csv"
        }
    )

    df = pd.read_csv(StringIO(response.content.decode("utf-8")), index_col=[0])
    df.insert(df.shape[1] - 1, "target1", df.pop("cat1"))
    df.insert(df.shape[1] - 1, "target2", df.pop("cat2"))
    _split_dataframe_and_save(df=df, dataset_name="right_heart_catheterization", directory=directory)


def _download_compass(
    directory: str = DEFAULT_DATA_DIR
) -> None:
    from io import StringIO
    from requests import get

    response = get(
        url="https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        params={
            "downloadformat": "csv"
        }
    )

    df = pd.read_csv(StringIO(response.content.decode("utf-8")))
    df = df.drop(columns=["id", "name", "first", "last"])
    df = df.rename(columns={
        "two_year_recid": "target"
    })
    _split_dataframe_and_save(df=df, dataset_name="compass", directory=directory)


def _download_mnist(
    directory: str = DEFAULT_DATA_DIR
) -> None:
    import gzip
    from tempfile import TemporaryDirectory

    from mnist import MNIST
    from requests import get
    from sklearn.model_selection import train_test_split

    train = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    test = ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

    with TemporaryDirectory() as _tmp_dir:
        tmp_dir = Path(_tmp_dir)
        for f in train + test:
            response = get(
                url=f"http://yann.lecun.com/exdb/mnist/{f}",
            )
            with (tmp_dir / f).open("wb") as fp:
                fp.write(response.content)
            with gzip.open((tmp_dir / f)) as gzip_fp:
                with (tmp_dir / f).with_suffix("").open("wb") as bin_fp:
                    bin_fp.write(gzip_fp.read())

        mnist = MNIST(_tmp_dir)

        test = mnist.load_testing()
        test_df = pd.DataFrame(
            data=[pixels + [label] for pixels, label in zip(test[0], test[1]) if label in (4, 9)],
            columns=[f"x{i}" for i in range(784)] + ["target"]
        )
        _save_dataframe(df=test_df, dataset_name=f"mnist_test", directory=directory)

        train_valid = mnist.load_training()
        train_valid_df = pd.DataFrame(
            data=[pixels + [label] for pixels, label in zip(train_valid[0], train_valid[1]) if label in (4, 9)],
            columns=[f"x{i}" for i in range(784)] + ["target"]
        )

        train, valid = train_test_split(
            train_valid_df,
            stratify=train_valid_df[["target"]],
            test_size=test_df.shape[0],
            random_state=RANDOM_SEED
        )
        _save_dataframe(df=train, dataset_name=f"mnist_train", directory=directory)
        _save_dataframe(df=valid, dataset_name=f"mnist_valid", directory=directory)


def _download_caltech(
    directory: str = DEFAULT_DATA_DIR
) -> None:
    from tempfile import TemporaryDirectory

    from torchvision.datasets import Caltech256
    from torchvision.models.resnet import resnet50, ResNet50_Weights

    resnet = resnet50(ResNet50_Weights.IMAGENET1K_V2)
    print(resnet)

    with TemporaryDirectory() as _tmp_dir:
        caltech = Caltech256(root=_tmp_dir, download=True)

        for image, label in caltech:
            print(x)
            break


def download_all(
    directory: str = DEFAULT_DATA_DIR
) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    # for download_function in [_download_caltech, _download_breast_cancer, _download_compass, _download_diabetes,
    #                            _download_mnist, _download_right_heart_catheterization]:
    for download_function in [_download_caltech]:
        download_function(directory=directory)


parser = ArgumentParser("Download datasets for running experiments.")
parser.add_argument("--directory", "-d", help="directory to store datasets")
if __name__ == "__main__":
    parser.parse_args()
    download_all()
