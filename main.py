from benchmark.benchmark_utils import (
    benchmark_knn_cnn,
    get_data_split,
    load_prepare_codon_data,
)
from cnn.cnn import CondensedNearestNeighbor
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


INPUT_DATA_PATH = "data/codon_usage.csv"


df, y = load_prepare_codon_data(INPUT_DATA_PATH)
X_train, X_test, y_train, y_test = get_data_split(df, y)

cnn = CondensedNearestNeighbor()


def run_benchmark() -> tuple([pd.DataFrame, pd.Series]):
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    X_reduced, y_reduced = benchmark_knn_cnn(
        knn, cnn, X_train, X_test, y_train, y_test
    )
    return X_reduced, y_reduced


def plot_results(X_reduced, y_reduced):
    X_reduced_codon = X_reduced
    y_reduced_codon = y_reduced

    cnn_codon_iters = cnn.iters.copy()
    cnn_codon_grabbag = cnn.grabbag.copy()

    cnn_codon_added = [X_train.shape[0] - g + 1 for g in cnn_codon_grabbag]

    print(
        f"Samples before: {X_train.shape[0]}, samples after: {X_reduced_codon.shape[0]}"
    )
    print(
        f"% of sampleset reduction: {((1-X_reduced_codon.shape[0]/X_train.shape[0])*100):.2f}"
    )

    _, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 6))
    ax.scatter(
        cnn_codon_iters,
        cnn_codon_added,
        label="Consistent (store) subset cardinality",
    )
    ax.scatter(
        cnn_codon_iters,
        cnn_codon_grabbag,
        label="Discard (grabbag) subset cardinality",
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Samples number")
    ax.set_yticks(np.arange(0, X_train.shape[0], step=1000))
    ax.set_xticks(np.arange(0, 4, step=1))
    ax.grid(which="major", linestyle=":")
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2)
    ax.set_title("Sample reduction during CNN steps")

    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("output/output.png", bbox_inches='tight')


def main():
    X_sampleset_reduced, y_sampleset_reduced = run_benchmark()
    plot_results(X_sampleset_reduced, y_sampleset_reduced)


if __name__ == "__main__":
    main()
