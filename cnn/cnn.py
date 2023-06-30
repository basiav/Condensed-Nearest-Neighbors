from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import warnings


# To ignore some annoying warnings
warnings.simplefilter(action="ignore", category=(FutureWarning, UserWarning))


DEFAULT_METRIC = "euclidean"


class CondensedNearestNeighbor:
    def __init__(
        self,
        *,
        n_neighbors=1,
        random_state=0,
        metric=DEFAULT_METRIC,
        n_jobs=-1,
    ):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.metric = metric
        self.n_jobs = n_jobs
        self.iters = []
        self.grabbag = []

    def update_iter_stats(self, it, len_idxs):
        self.iters.append(it)
        self.grabbag.append(len_idxs)

    def fit(self, X, y):
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self.rng = np.random.default_rng(self.random_state)
        self.original_samples_no = X.shape[0]
        return self

    def transform(self, X, y) -> tuple((pd.DataFrame, pd.Series)):
        # Store bin
        X_store = pd.DataFrame()
        y_store = pd.Series()

        # Keep the original dfs safe
        X = X.copy(deep=True)
        y = y.copy(deep=True)

        # Random_state in here
        idxs = np.array(X.index)
        self.rng.shuffle(idxs)
        self.update_iter_stats(0, len(idxs))

        # Samples to remove
        grabbag_idx = []

        # First sample always goes to store bin
        X_store = X_store.append(X.loc[idxs[0]].copy(deep=True))
        y_store = pd.concat(
            [y_store, pd.Series(y.loc[idxs[0]].copy(), index=[idxs[0]])]
        )

        # First pass through original sample set (training data)
        for i, loc_idx in enumerate(idxs):
            # We've already handled 1st sample
            if i == 0:
                continue

            x_sample = X.loc[loc_idx].values.reshape(1, -1)
            y_true = y.loc[loc_idx]

            self.knn.fit(X_store, y_store)
            y_pred = self.knn.predict(x_sample)

            # Misclassified case
            if y_pred != y_true:
                # Transfer to store bin
                X_store = X_store.append(X.loc[loc_idx].copy(deep=True))
                y_store = pd.concat(
                    [
                        y_store,
                        pd.Series(y.loc[loc_idx].copy(), index=[loc_idx]),
                    ]
                )

                X = X.drop(loc_idx)
                y = y.drop(loc_idx)

                grabbag_idx.append(i)

        idxs = np.delete(idxs, grabbag_idx)

        # Stop
        # 1st case: when the grabbag is exhausted
        # 2nd case: once one complete pass through grabbag
        # has been made without any sample removal
        one_clear_pass = False
        it = 0
        while len(idxs) > 0 and not one_clear_pass:
            self.rng.shuffle(idxs)
            grabbag_idx = []

            one_clear_pass = True
            for i, loc_idx in enumerate(idxs):
                x_sample = X.loc[loc_idx].values.reshape(1, -1)
                y_true = y.loc[loc_idx]

                self.knn.fit(X_store, y_store)
                y_pred = self.knn.predict(x_sample)

                # Misclassified case
                if y_pred != y_true:
                    # Transfer to store bin
                    X_store = X_store.append(X.loc[loc_idx].copy(deep=True))
                    y_store = pd.concat(
                        [
                            y_store,
                            pd.Series(y.loc[loc_idx].copy(), index=[loc_idx]),
                        ]
                    )

                    X = X.drop(loc_idx)
                    y = y.drop(loc_idx)

                    one_clear_pass = False
                    grabbag_idx.append(i)

            idxs = np.delete(idxs, grabbag_idx)
            self.update_iter_stats(it := it + 1, len(idxs))

        return X_store, y_store
