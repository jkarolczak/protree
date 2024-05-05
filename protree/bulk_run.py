import os

# ("breast_cancer", "diabetes", "compass", "rhc", "mnist", "caltech")

# SM-A
# ("sqrt", 0.33, 0.33, 7, "sqrt", "sqrt")
# (8, 3, 20, 12, 14, 16)

# SM-WA
# ("sqrt", "sqrt", 0.5, 0.5, "sqrt", 0.33)
# (8, 2, 20, 10, 11, 5)

# A-PETE
# ("sqrt", 0.33, 0.5, 7, 0.33, "sqrt)
# (0.05, 0.05, 0.01, 0.01, 0.05, 0.05)

# G-KM
# (0.7, 0.33, 0.5, 7, "sqrt", "sqrt")
# (3, 1, 8, 3, 7, 5)

# K-Means
# (4, 3, 5, 5, 10, 3)

if __name__ == "__main__":
    command_base = "python protree/experiment.py --log -t 1000 -d ./data "
    for p, k, dataset in zip(
            (0.7, 0.33, 0.5, 7, "sqrt", "sqrt"),
            (4, 3, 5, 5, 10, 3),
            ("breast_cancer", "diabetes", "compass", "rhc", "mnist", "caltech")
    ):
        command = command_base + f"-p {p} -k {k} {dataset}"
        print(command)
        os.system(command)
