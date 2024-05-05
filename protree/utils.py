def pprint_dict(dict_: dict, indent_level: int = 0) -> None:
    indent = "\t" * indent_level
    for key, value in dict_.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            pprint_dict(value, indent_level + 1)
        else:
            print(f"{indent}{key}: {value}")


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
