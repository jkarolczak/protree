from river import forest
from river.datasets import synth

from protree.explainers import APete

if __name__ == "__main__":
    dataset = synth.ConceptDriftStream(
        seed=42,
        position=500,
        width=10
    ).take(1000)

    x, y = list(zip(*list(dataset)))

    model = forest.ARFClassifier(seed=42, n_models=100, leaf_prediction="mc")

    for x_, y_ in zip(x, y):
        model.learn_one(x_, y_)

    explainer = APete(model, beta=0.05)
    protos = explainer.select_prototypes(list(x), list(y))
    for cls in protos:
        print(cls)
        for prot in protos[cls]:
            print(prot)
