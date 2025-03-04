import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import json
import os
import requests
from zipfile import ZipFile
import logging
import io

# Get the logging level from an environment variable, default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

test_set_settings = None

NEURPIS_DATA_URL = (
    "https://www.codabench.org/datasets/download/b9e59d0a-4db3-4da4-b1f8-3f609d1835b2/"
)

BLACK_SWAN_DATA_URL = (
    "https://www.codabench.org/datasets/download/3b3b3b3b-3b3b-3b3b-3b3b-3b3b3b3b3b3b/"
)


class Data:
    """
    A class to represent a dataset.

    Parameters:
        * input_dir (str): The directory path of the input data.

    Attributes:
        * __train_set (dict): A dictionary containing the train dataset.
        * __test_set (dict): A dictionary containing the test dataset.
        * input_dir (str): The directory path of the input data.

    Methods:
        * load_train_set(): Loads the train dataset.
        * load_test_set(): Loads the test dataset.
        * get_train_set(): Returns the train dataset.
        * get_test_set(): Returns the test dataset.
        * delete_train_set(): Deletes the train dataset.
        * get_syst_train_set(): Returns the train dataset with systematic variations.
    """

    def __init__(self, input_dir):
        """
        Constructs a Data object.

        Parameters:
            input_dir (str): The directory path of the input data.
        """

        self.__train_set = None
        self.__test_set = None
        self.input_dir = input_dir

    def load_train_set(self, sample_size=None, selected_indices=None):

        train_data_file = os.path.join(self.input_dir, "train", "data", "data.parquet")
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

        parquet_file = pq.ParquetFile(train_data_file)

        # Step 1: Determine the total number of rows
        total_rows = sum(
            parquet_file.metadata.row_group(i).num_rows
            for i in range(parquet_file.num_row_groups)
        )

        if sample_size is not None:
            if isinstance(sample_size, int):
                sample_size = min(sample_size, total_rows)
            elif isinstance(sample_size, float):
                if 0.0 <= sample_size <= 1.0:
                    sample_size = int(sample_size * total_rows)
                else:
                    raise ValueError("Sample size must be between 0.0 and 1.0")
            else:
                raise ValueError("Sample size must be an integer or a float")
        elif selected_indices is not None:
            if isinstance(selected_indices, list):
                selected_indices = np.array(selected_indices)
            elif isinstance(selected_indices, np.ndarray):
                pass
            else:
                raise ValueError("Selected indices must be a list or a numpy array")
            sample_size = len(selected_indices)
        else:
            sample_size = total_rows

        if selected_indices is None:
            selected_indices = np.random.choice(
                total_rows, size=sample_size, replace=False
            )

        selected_indices = np.sort(selected_indices)

        selected_indices_set = set(selected_indices)

        def get_sampled_data(data_file):
            selected_list = []
            with open(data_file, "r") as f:
                for i, line in enumerate(f):
                    # Check if the current line index is in the selected indices
                    if i not in selected_indices_set:
                        continue
                    if data_file.endswith(".detailed_labels"):
                        selected_list.append(line.strip())
                    else:
                        selected_list.append(float(line.strip()))
                    # Optional: stop early if all indices are found
                    if len(selected_list) == len(selected_indices):
                        break
            return np.array(selected_list)

        current_row = 0
        sampled_df = pd.DataFrame()
        for row_group_index in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_index).to_pandas()
            row_group_size = len(row_group)

            # Determine indices within the current row group that fall in the selected range
            within_group_indices = (
                selected_indices[
                    (selected_indices >= current_row)
                    & (selected_indices < current_row + row_group_size)
                ]
                - current_row
            )
            sampled_df = pd.concat(
                [sampled_df, row_group.iloc[within_group_indices]], ignore_index=True
            )

            # Update the current row count
            current_row += row_group_size

        selected_train_labels = get_sampled_data(train_labels_file)
        selected_train_weights = get_sampled_data(train_weights_file)
        selected_train_detailed_labels = get_sampled_data(train_detailed_labels_file)

        logger.info(f"Sampled train data shape: {sampled_df.shape}")
        logger.info(f"Sampled train labels shape: {selected_train_labels.shape}")
        logger.info(f"Sampled train weights shape: {selected_train_weights.shape}")
        logger.info(
            f"Sampled train detailed labels shape: {selected_train_detailed_labels.shape}"
        )

        self.__train_set = {
            "data": sampled_df,
            "labels": selected_train_labels,
            "total_rows": total_rows,
            "weights": selected_train_weights,
            "detailed_labels": selected_train_detailed_labels,
        }

        del (
            sampled_df,
            selected_train_labels,
            selected_train_weights,
            selected_train_detailed_labels,
        )

        buffer = io.StringIO()
        self.__train_set["data"].info(buf=buffer, memory_usage="deep", verbose=False)
        info_str = "Training Data :\n" + buffer.getvalue()
        logger.debug(info_str)
        logger.info("Train data loaded successfully")

    def load_test_set(self):

        test_data_dir = os.path.join(self.input_dir, "test", "data")

        # read test setting
        test_set = {
            "ztautau": pd.DataFrame(),
            "diboson": pd.DataFrame(),
            "ttbar": pd.DataFrame(),
            "htautau": pd.DataFrame(),
        }

        for key in test_set.keys():

            test_data_path = os.path.join(test_data_dir, f"{key}_data.parquet")
            test_set[key] = pd.read_parquet(test_data_path, engine="pyarrow")

        self.__test_set = test_set

        test_settings_file = os.path.join(
            self.input_dir, "test", "settings", "data.json"
        )
        with open(test_settings_file) as f:
            test_settings = json.load(f)

        self.ground_truth_mus = test_settings["ground_truth_mus"]

        for key in self.__test_set.keys():
            buffer = io.StringIO()
            self.__test_set[key].info(buf=buffer, memory_usage="deep", verbose=False)
            info_str = str(key) + ":\n" + buffer.getvalue()

            logger.debug(info_str)

        logger.info("Test data loaded successfully")

    def get_train_set(self):
        """
        Returns the train dataset.

        Returns:
            dict: The train dataset.
        """
        train_set = self.__train_set
        return train_set

    def get_test_set(self):
        """
        Returns the test dataset.

        Returns:
            dict: The test dataset.
        """
        return self.__test_set

    def delete_train_set(self):
        """
        Deletes the train dataset.
        """
        del self.__train_set


current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)


def __load_dataset(url):
    """
    Downloads and extracts the Neurips 2024 public dataset.

    Returns:
        Data: The path to the extracted input data.

    Raises:
        HTTPError: If there is an error while downloading the dataset.
        FileNotFoundError: If the downloaded dataset file is not found.
        zipfile.BadZipFile: If the downloaded file is not a valid zip file.
    """
    parent_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.dirname(parent_path)
    public_data_folder_path = os.path.join(current_path, "public_data")
    public_input_data_folder_path = os.path.join(
        current_path, "public_data", "input_data"
    )
    public_data_zip_path = os.path.join(current_path, "public_data.zip")

    # Check if public_data dir exists
    if os.path.isdir(public_data_folder_path):
        # Check if public_data/input_data dir exists
        if os.path.isdir(public_input_data_folder_path):
            return Data(public_input_data_folder_path)
        else:
            print("[!] public_data/input_dir directory not found")
    else:
        print("[!] public_data directory not found")

    # Check if public_data.zip exists
    if not os.path.isfile(public_data_zip_path):
        print("[!] public_data.zip does not exist")
        print("[*] Downloading public data, this may take few minutes")
        chunk_size = 1024 * 1024
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(public_data_zip_path, "wb") as file:
                # Iterate over the response in chunks
                for chunk in response.iter_content(chunk_size=chunk_size):
                    # Filter out keep-alive new chunks
                    if chunk:
                        file.write(chunk)

    # Extract public_data.zip
    print("[*] Extracting public_data.zip")
    with ZipFile(public_data_zip_path, "r") as zip_ref:
        zip_ref.extractall(public_data_folder_path)

    return Data(public_input_data_folder_path)


def Neurips2024_public_dataset():
    """
    Downloads and extracts the Neurips 2024 public dataset.

    Returns:
        Data: The path to the extracted input data.

    Raises:
        HTTPError: If there is an error while downloading the dataset.
        FileNotFoundError: If the downloaded dataset file is not found.
        zipfile.BadZipFile: If the downloaded file is not a valid zip file.
    """
    return __load_dataset(NEURPIS_DATA_URL)


def BlackSwan_public_dataset():
    """
    Downloads and extracts the Black swan public dataset.

    Returns:
        Data: The path to the extracted input data.

    Raises:
        HTTPError: If there is an error while downloading the dataset.
        FileNotFoundError: If the downloaded dataset file is not found.
        zipfile.BadZipFile: If the downloaded file is not a valid zip file.
    """
    return __load_dataset(BLACK_SWAN_DATA_URL)
