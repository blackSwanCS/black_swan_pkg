from HiggsML.score import Scoring
from HiggsML.datasets import download_dataset
import sys
import argparse
import pathlib
import os
import numpy as np
import json

from HiggsML.ingestion import Ingestion
from HiggsML.datasets import Data

root_dir_name = os.path.dirname(os.path.realpath(__file__))
working_dir = os.getcwd()

parser = argparse.ArgumentParser(
    description="This is script to run ingestion program for the competition"
)
parser.add_argument(
    "--model-type",
    "-m",
    help="Type of model in Model, sample ? BDT ? NN",
    default="sample_model",
)
parser.add_argument(
    "--output",
    "-o",
    help="Output file location",
    default=os.path.join(working_dir, "sample_result_submission"),
)
parser.add_argument(
    "--submission",
    "-s",
    help="Submission file location",
    default=os.path.join(working_dir, "sample_code_submission"),
)
parser.add_argument(
    "--codabench",
    help="True when running on Codabench",
    action="store_true",
)

parser.add_argument(
    "--systematics-tes",
    action="store_true",
    help="Whether to use tes systematics",
)
parser.add_argument(
    "--systematics-jes",
    action="store_true",
    help="Whether to use jes systematics",
)
parser.add_argument(
    "--systematics-soft-met",
    action="store_true",
    help="Whether to use soft_met systematics",
)
parser.add_argument(
    "--systematics-ttbar-scale",
    action="store_true",
    help="Whether to use ttbar_scale systematics",
)

parser.add_argument(
    "--systematics-diboson-scale",
    action="store_true",
    help="Whether to use diboson_scale systematics",
)

parser.add_argument(
    "--systematics-bkg-scale",
    action="store_true",
    help="Whether to use bkg_scale systematics",
)
parser.add_argument(
    "--num-pseudo-experiments",
    type=int,
    help="Number of pseudo experiments",
    default=25,
)
parser.add_argument(
    "--num-of-sets",
    type=int,
    help="Number of sets",
    default=25,
)


args = parser.parse_args()

if not args.codabench:
    output_dir = args.output
    submission_dir = args.submission
else:
    input_dir = "/app/input_data"
    output_dir = "/app/output"
    submission_dir = "/app/ingested_program"
    program_dir = "/app/program"


data = download_dataset(
    "blackSwan_data"
)  # change to "blackSwan_data" for the actual data

sys.path.append(submission_dir)


ingestion = Ingestion(data)

# Start timer
ingestion.start_timer()

from model import Model

# initialize submission
ingestion.init_submission(Model, model_type=args.model_type)

# fit submission
ingestion.fit_submission()
test_settings = {}
test_settings["systematics"] = {
    "tes": args.systematics_tes,
    "jes": args.systematics_jes,
    "soft_met": args.systematics_soft_met,
    "ttbar_scale": args.systematics_ttbar_scale,
    "diboson_scale": args.systematics_diboson_scale,
    "bkg_scale": args.systematics_bkg_scale,
}

test_settings["num_pseudo_experiments"] = args.num_pseudo_experiments
test_settings["num_of_sets"] = args.num_of_sets

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

test_settings["ground_truth_mus"] = (
    np.random.uniform(0.1, 3, test_settings["num_of_sets"])
).tolist()

random_settings_file = os.path.join(output_dir, "test_settings.json")
with open(random_settings_file, "w") as f:
    json.dump(test_settings, f)

# load test data
data.load_test_set()

# predict submission
ingestion.predict_submission(test_settings)

# compute result
ingestion.process_results_dict()

# save result
ingestion.save_result(output_dir)

# Stop timer
ingestion.stop_timer()

# Show duration
ingestion.show_duration()

# Save duration
ingestion.save_duration(output_dir)


# Init scoring
scoring = Scoring()

# Start timer
scoring.start_timer()

# Load ingestion duration
ingestion_duration_file = os.path.join(output_dir, "ingestion_duration.json")
scoring.load_ingestion_duration(ingestion_duration_file)


# Load ingestions results
scoring.load_ingestion_results(output_dir, output_dir)

# Compute Scores
scoring.compute_scores(test_settings)

# Write scores
scoring.write_scores()

# Stop timer
scoring.stop_timer()

print("\n----------------------------------------------")
print("[✔] Scoring Program executed successfully!")
print("----------------------------------------------\n\n")
