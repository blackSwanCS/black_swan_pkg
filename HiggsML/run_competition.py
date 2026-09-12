import sys
import argparse
from pathlib import Path
import os
import numpy as np
import json
import yaml

from analysisweb.sequencer import Sequencer
from analysisweb.plugin_loader import load_job_plugin

from .ingestion import Ingestion
from .score import Scoring
from .datasets import Data, download_dataset

working_dir = Path(os.getcwd())


def get_initial_entry(path):
    """Loads initial entry data from YAML config file."""

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    columns = config["columns"]
    initial_entry = {}
    for col in columns:
        if col["type"] != "action":
            initial_entry[col["key"]] = col["default"]

    return initial_entry


def main():

    parser = argparse.ArgumentParser(
        description="This is script to run ingestion program for the competition"
    )

    parser.add_argument(
        "--unique-date",
        type=str,
        default=os.getenv("UNIQUE_TIME_STAMP"),
        help="Unique date stamp for the model",
    )

    parser.add_argument(
        "--model-type",
        "-m",
        help="Type of model in Model, sample ? BDT ? NN",
        default=None,
    )

    parser.add_argument(
        "--input",
        "-i",
        help="Input file location",
        default=None,
    )

    parser.add_argument(
        "--submission",
        "-s",
        help="Submission file location",
        default=None,
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
        default=10,
    )
    parser.add_argument(
        "--num-of-sets",
        type=int,
        help="Number of sets",
        default=10,
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(working_dir) / "configs",
        help="Path to the main config files",
    )

    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path(working_dir) / "json",
        help="Path to the main config files",
    )

    args = parser.parse_args()

    models_dict = yaml.safe_load(
        (Path(args.config_path) / "models_paths.yaml").read_text()
    )

    if args.submission:
        submission_dir = Path(args.submission)
    elif args.model_type:
        submission_dir = Path(models_dict[args.model_type])
    else:
        submission_dir = Path(working_dir) / "sample_code_submission"

    if args.input is not None:
        data = Data(args.input)
    else:
        data = download_dataset("blackSwan_data")

    output_dir = Path(f"results/plots_{args.unique_date}")

    ingestion = Ingestion(data)

    # Start timer
    ingestion.start_timer()

    model = load_job_plugin(submission_dir / "model.py")

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

    initial_entry = get_initial_entry(Path(args.config_path) / "dashboard.yaml")
    initial_entry["date"] = args.unique_date
    
    ingestion_duration_file = os.path.join(output_dir, "ingestion_duration.json")

    print(initial_entry)

    sequencer = Sequencer(
        initial_entry=initial_entry,
        plots_dir=output_dir,
        json_dir=args.json_path,
    )

    sequencer.update({"Status": "Update table", "model": "Linear Regression"})

    sequencer.start()

    # initialize submission
    sequencer.add_algorithm(
        ingestion.init_submission,
        model.Model,
        model_type=args.model_type,
    )

    # fit submission
    sequencer.add_algorithm(
        ingestion.fit_submission,
    )

    # load test data
    sequencer.add_algorithm(
        data.load_test_set,
        aux=True,
    )

    # predict submission
    sequencer.add_algorithm(
        ingestion.predict_submission,
        test_settings=test_settings,
    )

    # compute result
    sequencer.add_algorithm(
        ingestion.process_results_dict,
    )

    # save result
    sequencer.add_algorithm(
        ingestion.save_result,
        output_dir=output_dir,
    )

    # Stop timer
    sequencer.add_algorithm(
        ingestion.stop_timer,
    )

    # Show duration
    sequencer.add_algorithm(
        ingestion.show_duration,
    )

    # Save duration
    sequencer.add_algorithm(
        ingestion.save_duration,
        output_dir=output_dir,
    )

    # Init scoring
    scoring = Scoring()

    # Start timer
    sequencer.add_algorithm(scoring.start_timer, aux=True)

    # Load ingestion duration
    sequencer.add_algorithm(
        scoring.load_ingestion_duration,
        ingestion_duration_file=ingestion_duration_file,
    )

    # Load ingestion results
    sequencer.add_algorithm(
        scoring.load_ingestion_results,
        prediction_dir=output_dir,
        score_dir=output_dir,
    )

    # Compute scores
    sequencer.add_algorithm(
        scoring.compute_scores,
        test_settings=test_settings,
    )

    sequencer.add_algorithm(
        scoring.save_figure,
        result_dir=output_dir
    )
    

    # Write scores
    sequencer.add_algorithm(
        scoring.write_scores,
    )

    # Stop timer
    sequencer.add_algorithm(scoring.stop_timer, aux=True)

    sequencer.print_sequence()
    sequencer.run()
    sequencer.end()


if __name__ == "__main__":
    main()
