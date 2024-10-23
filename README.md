[![PyPI ](https://img.shields.io/pypi/v/Protree)](https://pypi.org/project/protree/)
![Actions](https://github.com/jkarolczak/protree/actions/workflows/build_wheel.yml/badge.svg)

# Protree

Protree is a Python library containing a set of utilities for prototype selection for tree-based models. The implemented
tools include dataloaders, measures, prototype selection algorithms, and drift detection algorithms. The library is
designed to be used with the scikit-learn and river libraries.

The library was implemented as a part of master's thesis by [Jacek Karolczak](https://github.com/jkarolczak) under the
supervision of [prof. dr hab. Jerzy Stefanowski](https://scholar.google.pl/citations?user=id96GvIAAAAJ)
The main contribution of the thesis is the introduction of the measures to assess and compare prototypes, and the A-PETE
and ANCIENT algorithms.

## A-PETE: Adaptive Prototype Explanations of Tree Ensembles

A-PETE is a prototype selection method for ensemble of tree classifiers. A-PETE has been presented
at PP-RAI 2024: 5th Polish Conference on Artificial Intelligence and is scheduled to be published
in the conference proceedings.

### Abstract

The need for interpreting machine learning models is addressed through prototype explanations within the context of tree
ensembles. An algorithm named Adaptive Prototype Explanations of Tree Ensembles (A-PETE) is proposed to automatise the
selection of prototypes for these classifiers. Its unique characteristics is using a specialised distance measure and a
modified k-medoid approach. Experiments demonstrated its competitive predictive accuracy with respect to earlier
explanation algorithms. It also provides a sufficient number of prototypes for the purpose of interpreting the random
forest classifier.

### How to use?

Here is an example of how to use the A-PETE algorithm to select prototypes for a random forest classifier. The example
uses the Iris dataset and the random forest classifier from the scikit-learn library.

```py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from protree.explainers import APete

x, y = load_iris(as_frame=True, return_X_y=True)
random_forest = RandomForestClassifier().fit(x, y)
explainer = APete(model=random_forest)
prototypes = explainer.select_prototypes(x)
print(prototypes)
```

Output:

```
{
  0:   
         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
      0                5.1               3.5                1.4               0.2, 
  1:     
         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
     64                5.6               2.9                3.6               1.3, 
  2:
         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    104                6.5               3.0                5.8               2.2
}
```

### Using A-PETE? Cite us!

```bibtex
@article{karolczak2024apete,
  title={A-PETE: Adaptive Prototype Explanations of Tree Ensembles},
  author={Karolczak, Jacek and Stefanowski, Jerzy},
  journal={Progress in Polish Artificial Intelligence Research},
  volume={5},
  pages={2--8},
  year={2024},
  publisher={Warsaw University of Technology WUT Press}
}
```

## ANCIENT: Algorithm for New Concept Identification and Explanation in Tree-based models

ANCIENT is proposed to detect drift and also explain the nature of the drift using prototypes. The algorithm leverages
the principle that the measures describing data sub-populations from different distributions are more dissimilar than
those from the same distribution.

### How to use?

```py
from river import forest

from protree.data.stream_generators import Plane
from protree.explainers import APete
from protree.detectors import Ancient

window_size = 300

model = forest.ARFClassifier()
detector = Ancient(model=model, prototype_selector=APete, window_length=window_size, alpha=0.55,
                   measure="minimal_distance", strategy="total", clock=16)
ds = Plane(drift_position=[1150, 1800, 2500])

for i, (x, y) in enumerate(ds):
    model.learn_one(x, y)
    detector.update(x, y)
    if detector.drift_detected:
        print(f"{int(i - window_size / 2)}) Drift detected!")
```

Output:

```
1175) Drift detected!
1687) Drift detected!
1751) Drift detected!
2583) Drift detected!
```

## Experiments reproduction

The experiments conducted in the thesis can be reproduced using the scripts provided in the `scripts` directory,
specifically:

- `scripts/experiment-static.py` - allows to reproduce the experiments on static datasets, especially A-PETE
  effectiveness described in Chapter 3, as well as computing the measures described in Chapter 4,
- `scripts/experiment-steam-sklearn.py` - enables reproducing the experiments on the stream datasets proving the
  proposed measures' effectiveness in explaining drifts described in Chapter 5.2,
- `scripts/experiment-detect-drift.py` - makes it possible to reproduce the experiments on the stream datasets proving
  the ANCIENT algorithm's effectiveness in detecting drifts described in Chapter 5.3.

Usage of the scripts is described in the help message, which can be displayed by running the script with the `--help`
flag, for instance:

```shell
python scripts/experiment-static.py --help
```

Output:

```
Usage: experiment-static.py [OPTIONS]
                            {breast_cancer|caltech|compass|diabetes|mnist|rhc}
                            {KMeans|G_KM|SM_A|SM_WA|SG|APete}

Options:
  -d, --directory TEXT   Directory where datasets are stored.
  -p, --n_features TEXT  The number of features to consider when looking for
                         the best split. Allowable values are 'sqrt', positive
                         ints and floats between 0 and 1.
  -t, --n_trees INTEGER  Number of trees. Allowable values are positive ints.
  -kw, --kw_args TEXT    Additional, keyword arguments for the explainer. Must
                         be in the form of key=value,key2=value2...
  --log                  A flag indicating whether to log the results to
                         wandb.
  --help                 Show this message and exit.
```

For instance:

```shell
python scripts/experiment-static.py diabetes APete -p sqrt
```

Output:

```
total_n_prototypes: 5
score/accuracy/train/random_forest: 1.0
score/accuracy/train/prototypes: 0.8391304347826087
score/accuracy/valid/random_forest: 0.7532467532467533
score/accuracy/valid/prototypes: 0.7532467532467533
score/accuracy/test/random_forest: 0.7337662337662338
score/accuracy/test/prototypes: 0.7142857142857143
score/gmean/train/random_forest: 1.0
score/gmean/train/prototypes: 0.8391304347826087
score/gmean/valid/random_forest: 0.7532467532467533
score/gmean/valid/prototypes: 0.7532467532467533
score/gmean/test/random_forest: 0.7337662337662337
score/gmean/test/prototypes: 0.7142857142857143
score/valid/fidelity: 0.948051948051948
score/valid/hubness: 0.9428732702115488
score/valid/mean_in_distribution: 0.09985377777777778
score/valid/mean_out_distribution: 0.01757222222222222
vector/valid/partial_in_distribution:
        0: [0.2089, 0.10524, 0.04924]
        1: [0.07157407407407407, 0.06431481481481481]
vector/valid/partial_hubnesses:
        0: 0.8907627072209605
        1: 0.9949838332021369
vector/valid/partial_out_distribution:
        0: [0.02162962962962963, 0.017, 0.02248148148148148]
        1: [0.01521, 0.01154]
vector/valid/consistent_votes:
        0: [0.9365079365079365, 1.0, 0.9047619047619048]
        1: [0.9230769230769231, 1.0]
vector/valid/voting_frequency:
        0: [0.4090909090909091, 0.16883116883116883, 0.13636363636363635]
        1: [0.16883116883116883, 0.11688311688311688]
```