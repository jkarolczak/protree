import os

if __name__ == "__main__":
    command_base = "python protree/experiment.py -t 1000 --log -d ./data "
    for dataset in ("breast_cancer", "compass", "diabetes", "rhc", "mnist", "caltech"):
        for k in range(1, 11):
            for p in ("sqrt", 0.33, 0.5, 0.7, 7):
                os.system(
                    command_base + f"-p {p} -k {k} {dataset}"
                )
