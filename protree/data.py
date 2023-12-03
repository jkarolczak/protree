from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Literal

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


def _download_rhs(
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
    _split_dataframe_and_save(df=df, dataset_name="rhs", directory=directory)


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

    import torch
    from torchvision import transforms
    from torchvision.datasets import Caltech256
    from torchvision.models.resnet import resnet50, ResNet50_Weights

    def collate_fn(batch: list[tuple[torch.Tensor, int]]):
        images = []
        labels = []
        for idx, (image, label) in enumerate(batch):
            images.append(image.repeat(3, 1, 1) if image.shape[0] == 1 else image)
            labels.append(label)
        return torch.stack(images, dim=0), labels

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    resnet = resnet50(ResNet50_Weights.IMAGENET1K_V2).to(device)

    with TemporaryDirectory() as _tmp_dir:
        caltech = Caltech256(root=_tmp_dir, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(
            caltech,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn
        )

        x = []
        y = []

        for i, (images, labels) in enumerate(dataloader):
            vector = resnet(images.to(device))
            x.extend(vector.cpu().tolist())
            y.extend(labels)

        df = pd.DataFrame(data=x, columns=[f"x{i + 1}" for i in range(len(x[0]))])
        df["target"] = y

    _split_dataframe_and_save(df=df, dataset_name="caltech", directory=directory)


dataset_func_mapping = {
    "breast_cancer": _download_breast_cancer,
    "caltech": _download_caltech,
    "compass": _download_compass,
    "diabetes": _download_diabetes,
    "mnist": _download_mnist,
    "rhs": _download_rhs
}


def download_all(
    directory: str = DEFAULT_DATA_DIR,
    dataset_names: list[Literal["all", "breast_cancer", "caltech", "compass", "diabetes", "mnist", "rhs"]] = "all",
    verbose: bool = True
) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    allowable_datasets = ["breast_cancer", "caltech", "compass", "diabetes", "mnist", "rhs"]

    if "all" in dataset_names:
        dataset_names = allowable_datasets

    for dataset_name in dataset_names:
        if dataset_name not in allowable_datasets:
            raise NameError(f"Unknown dataset: {dataset_name}")
        if verbose:
            print(f"Processing {dataset_name}...")
        globals()[f"_download_{dataset_name}"](directory=directory)


parser = ArgumentParser("Download datasets for running experiments.")
parser.add_argument("--directory", "-d", default=DEFAULT_DATA_DIR, help="Directory to store datasets")
parser.add_argument("--silent", "-s", action=BooleanOptionalAction,
                    help="Supress displaying progress.")
parser.add_argument("--dataset-names", "-n", default="all", help="Comma-separated list of dataset names to"
                                                                 " download. Allowable values are 'breast_cancer', "
                                                                 "'caltech', 'compass', 'diabetes', 'mnist' and 'rhs'. "
                                                                 "Use 'all' to download all datasets")

if __name__ == "__main__":
    args = parser.parse_args()

    download_all(
        directory=args.directory,
        dataset_names=[s.strip() for s in args.dataset_names.split(",")],
        verbose=not args.silent
    )
