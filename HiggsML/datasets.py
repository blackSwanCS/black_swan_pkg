import numpy as np
import pandas as pd
import json
import os
import subprocess
import wget


test_set_settings = None


class Data:

    def __init__(self, input_dir, data_format="csv"):

        self.__train_set = None
        self.__test_set = None
        self.data_format = data_format
        self.input_dir = input_dir

    def load_train_set(self):
        print("[*] Loading Train data")

        if self.data_format == "csv":
            train_data_file = os.path.join(self.input_dir, "train", "data", "data.csv")
        if self.data_format == "parquet":
            train_data_file = os.path.join(
                self.input_dir, "train", "data", "data.parquet"
            )
        train_labels_file = os.path.join(
            self.input_dir, "train", "labels", "data.labels"
        )
        train_settings_file = os.path.join(
            self.input_dir, "train", "settings", "data.json"
        )
        train_weights_file = os.path.join(
            self.input_dir, "train", "weights", "data.weights"
        )
        train_detailed_labels_file = os.path.join(
            self.input_dir, "train", "detailed_labels", "data.detailed_labels"
        )

        # read train labels
        with open(train_labels_file, "r") as f:
            train_labels = np.array(f.read().splitlines(), dtype=float)

        # read train settings
        with open(train_settings_file) as f:
            train_settings = json.load(f)

        # read train weights
        with open(train_weights_file) as f:
            train_weights = np.array(f.read().splitlines(), dtype=float)

        # read train process flags
        with open(train_detailed_labels_file) as f:
            train_detailed_labels = f.read().splitlines()

        if self.data_format == "parquet":
            self.__train_set = {
                "data": pd.read_parquet(train_data_file, engine="pyarrow"),
                "labels": train_labels,
                "settings": train_settings,
                "weights": train_weights,
                "detailed_labels": train_detailed_labels,
            }

        else:
            self.__train_set = {
                "data": pd.read_csv(train_data_file),
                "labels": train_labels,
                "settings": train_settings,
                "weights": train_weights,
                "detailed_labels": train_detailed_labels,
            }

        del train_labels, train_settings, train_weights, train_detailed_labels

        print(self.__train_set["data"].info(verbose=False, memory_usage="deep"))

        test_settings_file = os.path.join(
            self.input_dir, "test", "settings", "data.json"
        )
        with open(test_settings_file) as f:
            test_settings = json.load(f)

        self.ground_truth_mus = test_settings["ground_truth_mus"]

        print("[*] Train data loaded successfully")

    def load_test_set(self):
        print("[*] Loading Test data")

        test_data_dir = os.path.join(self.input_dir, "test", "data")

        # read test setting

        test_set = {
            "ztautau": pd.DataFrame(),
            "diboson": pd.DataFrame(),
            "ttbar": pd.DataFrame(),
            "htautau": pd.DataFrame(),
        }

        for key in test_set.keys():
            if self.data_format == "csv":
                test_data_path = os.path.join(test_data_dir, f"{key}_data.csv")
                test_set[key] = pd.read_csv(test_data_path)

            elif self.data_format == "parquet":
                test_data_path = os.path.join(test_data_dir, f"{key}_data.parquet")
                test_set[key] = pd.read_parquet(test_data_path, engine="pyarrow")

        self.__test_set = test_set

        print("[*] Test data loaded successfully")

    def generate_psuedo_exp_data(
        self,
        set_mu=1,
        tes=1.0,
        jes=1.0,
        soft_met=1.0,
        w_scale=None,
        bkg_scale=None,
        seed=42,
    ):
        from HiggsML.systematics import get_bootstraped_dataset, get_systematics_dataset

        # get bootstrapped dataset from the original test set
        pesudo_exp_data = get_bootstraped_dataset(
            self.__test_set,
            mu=set_mu,
            w_scale=w_scale,
            bkg_scale=bkg_scale,
            seed=seed,
        )
        test_set = get_systematics_dataset(
            pesudo_exp_data,
            tes=tes,
            jes=jes,
            soft_met=soft_met,
        )

        return test_set

    def get_train_set(self):
        return self.__train_set

    def delete_train_set(self):
        del self.__train_set

    def get_syst_train_set(
        self, tes=1.0, jes=1.0, soft_met=1.0, w_scale=None, bkg_scale=None
    ):
        from HiggsML.systematics import systematics

        if self.__train_set is None:
            self.load_train_set()
        return systematics(self.__train_set, tes, jes, soft_met, w_scale, bkg_scale)


def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):
    data = data_set["data"].copy()
    train_set = {}
    test_set = {}
    full_size = len(data)
    np.random.seed(random_state)
    if isinstance(test_size, float):
        test_number = int(test_size * full_size)
        random_index = np.random.randint(0, full_size, test_number)
    elif isinstance(test_size, int):
        random_index = np.random.randint(0, full_size, test_size)
    else:
        raise ValueError("test_size should be either float or int")

    remaining_index = np.setdiff1d(np.arange(full_size), random_index)

    for key in data_set.keys():
        if (key != "data") or (key != "settings"):
            array = np.array(data_set[key])
            test_set[key] = array[random_index]
            train_set[key] = array[remaining_index]

    test_set["data"] = data.iloc[random_index]
    train_set["data"] = data.iloc[remaining_index]

    if reweight is True:
        signal_weight = np.sum(data_set["weights"][data_set["labels"] == 1])
        background_weight = np.sum(data_set["weights"][data_set["labels"] == 0])
        signal_weight_train = np.sum(train_set["weights"][train_set["labels"] == 1])
        background_weight_train = np.sum(train_set["weights"][train_set["labels"] == 0])
        signal_weight_test = np.sum(test_set["weights"][test_set["labels"] == 1])
        background_weight_test = np.sum(test_set["weights"][test_set["labels"] == 0])

        train_set["weights"][train_set["labels"] == 1] = train_set["weights"][
            train_set["labels"] == 1
        ] * (signal_weight / signal_weight_train)
        test_set["weights"][test_set["labels"] == 1] = test_set["weights"][
            test_set["labels"] == 1
        ] * (signal_weight / signal_weight_test)

        train_set["weights"][train_set["labels"] == 0] = train_set["weights"][
            train_set["labels"] == 0
        ] * (background_weight / background_weight_train)
        test_set["weights"][test_set["labels"] == 0] = test_set["weights"][
            test_set["labels"] == 0
        ] * (background_weight / background_weight_test)

    return train_set, test_set


def reweight(data_set):

    from HiggsML.systematics import LHC_NUMBERS

    for key in LHC_NUMBERS.keys():
        detailed_label = np.array(data_set["detailed_labels"])
        weight_key = np.sum(data_set["weights"][detailed_label == key])
        data_set["weights"][detailed_label == key] = data_set["weights"][
            detailed_label == key
        ] * (LHC_NUMBERS[key] / weight_key)

        print(f"Reweighting {key} with {LHC_NUMBERS[key] / weight_key}")
        print(
            f"New weight for {key} is {np.sum(data_set['weights'][detailed_label == key])}"
        )
    return data_set


# Datasets


def Neurips2024_public_dataset():
    current_path = os.getcwd()
    file_read_loc = os.path.join(current_path, "public_data")
    if not os.path.isdir(file_read_loc):
        os.mkdir(file_read_loc)

    file = "public_data.zip"
    if file not in os.listdir(file_read_loc):
        wget.download(
            "https://www.codabench.org/datasets/download/7c48062b-6500-4d9e-a899-9f1913ebb5ff/",
            out=os.path.join(file_read_loc, "public_data.zip"),
        )

    if "input_data" not in os.listdir(file_read_loc):
        subprocess.run(
            ["unzip", os.path.join(file_read_loc, file), "-d", file_read_loc]
        )

    return Data(
        os.path.join(current_path, "public_data", "input_data"), data_format="parquet"
    )

    # 71501d1e-3e41-4c63-8094-8ac657728fda


def BlackSwan_public_dataset():
    current_path = os.getcwd()
    file_read_loc = os.path.join(current_path, "public_data")
    if not os.path.isdir(file_read_loc):
        os.mkdir(file_read_loc)

    file = "public_data.zip"
    if file not in os.listdir(file_read_loc):
        wget.download(
            "https://www.codabench.org/datasets/download/7c48062b-6500-4d9e-a899-9f1913ebb5ff/",
            out=os.path.join(file_read_loc, "public_data.zip"),
        )

    if "input_data" not in os.listdir(file_read_loc):
        subprocess.run(
            ["unzip", os.path.join(file_read_loc, file), "-d", file_read_loc]
        )

    return Data(
        os.path.join(current_path, "public_data", "input_data"), data_format="parquet"
    )
