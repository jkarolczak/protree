from river import forest
from river.datasets import synth

from protree.explainers import APete
from protree.metrics.compare import mutual_information, mean_minimal_distance
from protree.metrics.group import fidelity_with_model, entropy_hubness, contribution, mean_in_distribution, \
    mean_out_distribution

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
    prototypes = explainer.select_prototypes(list(x), list(y))

    statistics = {
        "n prototypes": sum([len(c) for c in prototypes.values()]),
        "score/valid/prototypes": explainer.score_with_prototypes(list(x), list(y), prototypes),
        "score/train/fidelity (with model)": fidelity_with_model(prototypes, explainer, list(x)),
        "score/train/contribution": contribution(prototypes, explainer, list(x)),
        "score/train/hubness": entropy_hubness(prototypes, explainer, list(x), list(y)),
        "score/train/mean_in_distribution": mean_in_distribution(prototypes, explainer, list(x), list(y)),
        "score/train/mean_out_distribution": mean_out_distribution(prototypes, explainer, list(x), list(y))
    }

    print(statistics)

    print(mutual_information(prototypes, prototypes, list(x)))
    print(mean_minimal_distance(prototypes, prototypes))
