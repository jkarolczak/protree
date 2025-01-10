import arff
import pandas as pd
from river.drift import PageHinkley


def load_arff_acc(path: str) -> pd.DataFrame:
    data = arff.load(path)

    data = [list(map(lambda x: x if x is not None else 0, row)) for row in data]

    df = pd.DataFrame(data)
    return df


def detect_drift(dataframe: pd.DataFrame) -> list[int]:
    detector = PageHinkley(delta=0.01, threshold=.3, alpha=0.997)
    drift_positions = []

    for idx, row in dataframe.iterrows():
        detector.update(row.values[-1])

        if idx > 40000 and detector.drift_detected:
            drift_positions.append(idx)
            print(f"Drift detected at stream position: {idx}")

    return drift_positions


if __name__ == "__main__":
    dataset_path = "data_arff/plane100.arff"
    df_acc = load_arff_acc(dataset_path)

    drift_positions = detect_drift(df_acc)

    print("Drift positions:", drift_positions)
