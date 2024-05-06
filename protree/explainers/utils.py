from typing import Dict, Any, Type

import numpy as np
import pandas as pd


def _1d_np_to_2d_np(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x, axis=0)


def parse_input(x: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(x, list) and len(x) == 1:
        return np.array(x)
    match type(x):
        case np.ndarray:
            match len(x.shape):
                case 1:
                    return _1d_np_to_2d_np(x)
                case 2:
                    return x
                case _:
                    raise ValueError("Invalid input size.")
        case pd.Series:
            return _1d_np_to_2d_np(x.to_numpy())
        case pd.DataFrame:
            return x.to_numpy()
        case _:
            raise TypeError(f"Invalid input type: {type(x)}.")


def _type_to_np_dtype(x: Dict[Any, Any]) -> Type | str:
    val = list(x.keys())[0]
    if isinstance(val, str):
        return "<U50"
    return type(val)
