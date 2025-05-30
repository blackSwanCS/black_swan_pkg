# ------------------------------------------
# Imports
# ------------------------------------------
import numpy as np
import os
from datetime import datetime as dt
import json
from itertools import product
import logging

log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

DEFAULT_INGESTION_SEED = 31415

from HiggsML.systematics import generate_pseudo_exp_data

# ------------------------------------------
# Ingestion Class
# ------------------------------------------


class Ingestion:
    """
    Class for handling the ingestion process.

    Args:
        data (object): The data object.

    Attributes:
        * start_time (datetime): The start time of the ingestion process.
        * end_time (datetime): The end time of the ingestion process.
        * model (object): The model object.
        * data (object): The data object.
    """

    def __init__(self, data=None):
        """
        Initialize the Ingestion class.

        Args:
            data (object): The data object.
        """
        self.start_time = None
        self.end_time = None
        self.model = None
        self.data = data

    def start_timer(self):
        """
        Start the timer for the ingestion process.
        """
        self.start_time = dt.now()

    def stop_timer(self):
        """
        Stop the timer for the ingestion process.
        """
        self.end_time = dt.now()

    def get_duration(self):
        """
        Get the duration of the ingestion process.

        Returns:
            timedelta: The duration of the ingestion process.
        """
        if self.start_time is None:
            logger.warning("Timer was never started. Returning None")
            return None

        if self.end_time is None:
            logger.warning("Timer was never stopped. Returning None")
            return None

        return self.end_time - self.start_time

    def show_duration(self):
        """
        Show the duration of the ingestion process.
        """
        print("\n---------------------------------")
        print(f"[✔] Total duration: {self.get_duration()}")
        print("---------------------------------")

    def save_duration(self, output_dir=None):
        """
        Save the duration of the ingestion process to a file.

        Args:
            output_dir (str): The output directory to save the duration file.
        """
        duration = self.get_duration()
        duration_in_mins = int(duration.total_seconds() / 60)
        duration_file = os.path.join(output_dir, "ingestion_duration.json")
        if duration is not None:
            with open(duration_file, "w") as f:
                f.write(json.dumps({"ingestion_duration": duration_in_mins}, indent=4))

    def load_train_set(self, **kwargs):
        """
        Load the training set.

        Returns:
            object: The loaded training set.
        """
        self.data.load_train_set(**kwargs)
        return self.data.get_train_set()

    def init_submission(self, Model, model_type="sample_model"):
        """
        Initialize the submitted model.

        Args:
            Model (object): The model class.
        """
        logger.info("Initializing Submmited Model")
        from HiggsML.systematics import systematics

        self.model = Model(
            get_train_set=self.load_train_set,
            systematics=systematics,
            model_type=model_type,
        )
        self.data.delete_train_set()

    def fit_submission(self):
        """
        Fit the submitted model.
        """
        logger.info("Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self, test_settings, initial_seed=DEFAULT_INGESTION_SEED):
        """
        Make predictions using the submitted model.

        Args:
            test_settings (dict): The test settings.
        """
        logger.info(
            "Calling predict method of submitted model with seed: %s", initial_seed
        )

        dict_systematics = test_settings["systematics"]
        num_pseudo_experiments = test_settings["num_pseudo_experiments"]
        num_of_sets = test_settings["num_of_sets"]

        # get set indices
        set_indices = np.arange(0, num_of_sets)
        # get test set indices per set
        test_set_indices = np.arange(0, num_pseudo_experiments)

        # create a product of set and test set indices all combinations of tuples
        all_combinations = list(product(set_indices, test_set_indices))
        # randomly shuffle all combinations of indices
        random_state_initial = np.random.RandomState(initial_seed)
        random_state_initial.shuffle(all_combinations)

        full_test_set = self.data.get_test_set()
        del self.data

        self.results_dict = {}
        for set_index, test_set_index in all_combinations:

            # create a seed
            seed = (set_index * num_pseudo_experiments) + test_set_index + initial_seed

            # get mu value of set from test settings
            set_mu = test_settings["ground_truth_mus"][set_index]

            test_set = generate_pseudo_exp_data(
                full_test_set, set_mu, dict_systematics, seed
            )

            logger.debug(
                f"set_index: {set_index} - test_set_index: {test_set_index} - seed: {seed}"
            )

            predicted_dict = self.model.predict(test_set)
            predicted_dict["test_set_index"] = test_set_index

            if set_index not in self.results_dict:
                self.results_dict[set_index] = []
            self.results_dict[set_index].append(predicted_dict)

    def process_results_dict(self):
        # loop over sets
        for key in self.results_dict.keys():
            set_result = self.results_dict[key]

            # Sort the list of dictionaries by "test_set_index" if it exists
            if (
                set_result
                and isinstance(set_result, list)
                and "test_set_index" in set_result[0]
            ):
                set_result.sort(key=lambda x: x["test_set_index"])

            # Initialize a dictionary to store all extracted fields
            ingestion_result_dict = {}

            # Extract all available fields from the first test_set_dict
            if set_result and isinstance(set_result, list) and len(set_result) > 0:
                # Get all possible keys from the first dictionary (assuming all have same keys)
                available_keys = set_result[0].keys()

                # For each key, collect all values across test_set_dicts
                for field in available_keys:
                    if field != "test_set_index":  # Skip the sorting key
                        field_values = [
                            test_set_dict[field] for test_set_dict in set_result
                        ]
                        ingestion_result_dict[
                            field + "s" if not field.endswith("s") else field
                        ] = field_values

            self.results_dict[key] = ingestion_result_dict

    def save_result(self, output_dir=None):
        """
        Save the ingestion result to files.

        Args:
            output_dir (str): The output directory to save the result files.
        """
        for key in self.results_dict.keys():
            result_file = os.path.join(output_dir, "result_" + str(key) + ".json")
            with open(result_file, "w") as f:
                f.write(json.dumps(self.results_dict[key], indent=4))
