import pandas as pd
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import f1_score
import numpy as np

from termcolor import colored
from sklearn.metrics import roc_auc_score


def load_prepare_codon_data(
    input_data_path: str,
) -> tuple([pd.DataFrame, pd.Series]):
    df = pd.read_csv(input_data_path)
    df = df[pd.to_numeric(df["UUU"], errors="coerce").notnull()].copy()

    df = df.copy()  # to avoid irritating SettingWithCopyWarning
    df["UUU"] = df.loc[:, "UUU"].astype(float)
    df["UUC"] = df.loc[:, "UUC"].astype(float)

    df = df.loc[~df["Ncodons"] < 1000, :]
    df = df.loc[df["Kingdom"] != "plm", :]
    df = df.drop(
        ["DNAtype", "SpeciesID", "Ncodons", "SpeciesName"], axis="columns"
    )

    kingdom_mapping = {
        "arc": 0,
        "bct": 1,
        "pln": 2,
        "inv": 2,
        "vrt": 2,
        "mam": 2,
        "rod": 2,
        "pri": 2,
        "phg": 3,
        "vrl": 4,
    }
    df = df.replace({"Kingdom": kingdom_mapping})
    y = df.pop("Kingdom")

    return df, y


def get_data_split(
    df: pd.DataFrame, y: pd.Series
) -> tuple([pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]):
    return train_test_split(df, y, test_size=0.2, random_state=0, stratify=y)


def benchmark_knn_cnn(
    knn, cnn, X_train, X_test, y_train, y_test, auroc=False
) -> tuple((pd.DataFrame, pd.Series)):
    # Training
    # KNN 1
    start_time = time()
    knn.fit(X_train, y_train)
    end_time = time()

    knn_fit_time = end_time - start_time

    # (CNN) KNN 2
    cnn.fit(X_train, y_train)
    # All the magic happens in .transform()
    start_time = time()
    X_reduced, y_reduced = cnn.transform(X_train, y_train)
    end_time = time()

    cnn_transform_time = end_time - start_time

    # Now, as we've got reduced X and y
    # the real CNN classifier benchmark begins
    # CNN training
    start_time = time()
    cnn.fit(X_reduced, y_reduced)
    cnn.knn.fit(X_reduced, y_reduced)
    end_time = time()

    cnn_fit_time = end_time - start_time

    # Prediction
    # KNN
    start_time = time()
    y_pred_knn_test = knn.predict(np.array(X_test, dtype=float))
    knn_predict_time = end_time - start_time

    knn_predict_time = end_time - start_time

    # CNN
    start_time = time()
    y_pred_cnn_test = cnn.knn.predict(np.array(X_test, dtype=float))
    end_time = time()

    cnn_predict_test_time = end_time - start_time

    # CNN safety check - prediction of training data
    start_time = time()
    y_pred_cnn_train = cnn.knn.predict(np.array(X_train, dtype=float))
    end_time = time()

    cnn_predict_train_time = end_time - start_time

    f1_cnn_train = f1_score(y_train, y_pred_cnn_train, average="macro")

    print(colored(f"{'-'*20} CNN SAFETY CHECK {'-'*20}\n", "blue"))
    print(
        f"CNN prediction on training data is the same as on \
    original KNN before sample reduction"
    )
    print(
        f"y_pred_cnn_train == y_train).all(): \
    {(y_pred_cnn_train == y_train).all()}"
    )
    print(
        f"CNN score on training dataset: {cnn.knn.score(np.array(X_train, dtype=float), np.array(y_train, dtype=float))}"
    )
    print(
        f"CNN score on reduced dim dataset: {cnn.knn.score(np.array(X_reduced, dtype=float), np.array(y_reduced, dtype=float))}"
    )
    print(f"CNN f1 score for training dataset: {f1_cnn_train}")
    print()

    f1_knn = f1_score(y_test, y_pred_knn_test, average="macro")
    f1_cnn = f1_score(y_test, y_pred_cnn_test, average="macro")

    print(colored(f"{'-'*20} TIMES {'-'*20}\n", "blue"))
    print(f"KNN fit time: {knn_fit_time:.2f}")
    print(f"CNN fit time: {cnn_fit_time:.2f}")
    print(f"CNN actual training (transform) time: {cnn_transform_time:.2f}")
    print()
    print(f"KNN prediction time: {knn_predict_time:.2f}")
    print(f"CNN prediction time: {cnn_predict_test_time:.2f}")
    print()

    print(colored(f"{'-'*20} ACCURACY {'-'*20}\n", "blue"))
    print(f"KNN F1 score: {100 * f1_knn:.2f}%")
    print(f"CNN F1: {100 * f1_cnn:.2f}%")

    # Only for binary classes datasets
    if auroc:
        # KNN
        start_time = time()
        knn_y_pred_proba = knn.predict_proba(X_test)
        end_time = time()

        knn_predict_proba_time = end_time - start_time

        # CNN
        start_time = time()
        cnn_y_pred_proba = cnn.knn.predict_proba(X_test)
        end_time = time()

        cnn_predict_proba_time = end_time - start_time

        knn_auroc = roc_auc_score(y_test, knn_y_pred_proba[:, 1])
        cnn_auroc = roc_auc_score(y_test, cnn_y_pred_proba[:, 1])

        print(colored(f"{'-'*20} AUROC {'-'*20}\n", "blue"))
        print(f"KNN predict_proba time: {knn_predict_proba_time}")
        print(f"KNN AUROC: {100 * knn_auroc:.2f}%")
        print(f"CNN predict_proba time: {cnn_predict_proba_time}")
        print(f"CNN AUROC: {100 * cnn_auroc:.2f}%")

    return X_reduced, y_reduced
