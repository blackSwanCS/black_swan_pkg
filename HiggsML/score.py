# ------------------------------------------
# Imports
# ------------------------------------------

import json
import logging
import os
import sys
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import mpld3
import argparse

log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)


class Scoring:
    """
    This class is used to compute the scores for the competition.
    For more details, see the :doc:`evaluation page <../pages/evaluation>`.

    Atributes:
        * start_time (datetime): The start time of the scoring process.
        * end_time (datetime): The end time of the scoring process.
        * ingestion_results (list): The ingestion results.
        * ingestion_duration (float): The ingestion duration.
        * scores_dict (dict): The scores dictionary.

    Methods:
        * start_timer(): Start the timer.
        * stop_timer(): Stop the timer.
        * get_duration(): Get the duration of the scoring process.
        * show_duration(): Show the duration of the scoring process.
        * load_ingestion_duration(ingestion_duration_file): Load the ingestion duration.
        * load_ingestion_results(prediction_dir="./",score_dir="./"): Load the ingestion results.
        * compute_scores(test_settings): Compute the scores.
        * RMSE_score(mu, mu_hat, delta_mu_hat): Compute the RMSE score.
        * MAE_score(mu, mu_hat, delta_mu_hat): Compute the MAE score.
        * Quantiles_Score(mu, p16, p84, eps=1e-3): Compute the Quantiles Score.
        * write_scores(): Write the scores.
        * save_figure(mu, p16s, p84s, set=0): Save the figure.

    """

    def __init__(self):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.ingestion_results = None
        self.ingestion_duration = None

        self.scores_dict = {}

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            logger.warning("Timer was never started. Returning None")
            return None

        if self.end_time is None:
            logger.warning("Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def load_ingestion_duration(self, ingestion_duration_file):
        """
        Load the ingestion duration.

        Args:
            ingestion_duration_file (str): The ingestion duration file.
        """
        logger.info(f"Reading ingestion duration from {ingestion_duration_file}")

        try:
            with open(ingestion_duration_file) as f:
                self.ingestion_duration = json.load(f)["ingestion_duration"]
        except FileNotFoundError:
            logger.warning(
                f"File {ingestion_duration_file} not found. Setting ingestion duration to 0."
            )
            self.ingestion_duration = 0
        except json.JSONDecodeError:
            logger.warning(
                f"File {ingestion_duration_file} is not a valid JSON file. Setting ingestion duration to 0."
            )
            self.ingestion_duration = 0
        except Exception as e:
            logger.warning(
                f"Error reading file {ingestion_duration_file}: {e}. Setting ingestion duration to 0."
            )
            self.ingestion_duration = 0

    def load_ingestion_results(self, prediction_dir="./", score_dir="./"):
        """
        Load the ingestion results.

        Args:
            prediction_dir (str, optional): location of the predictions. Defaults to "./".
            score_dir (str, optional): location of the scores. Defaults to "./".
        """
        ingestion_results_with_set_index = []
        # loop over sets (1 set = 1 value of mu)
        for file in os.listdir(prediction_dir):
            if file.startswith("result_"):
                set_index = int(
                    file.split("_")[1].split(".")[0]
                )  # file format: result_{set_index}.json
                results_file = os.path.join(prediction_dir, file)
                with open(results_file) as f:
                    ingestion_results_with_set_index.append(
                        {"set_index": set_index, "results": json.load(f)}
                    )
        ingestion_results_with_set_index = sorted(
            ingestion_results_with_set_index, key=lambda x: x["set_index"]
        )
        self.ingestion_results = [
            x["results"] for x in ingestion_results_with_set_index
        ]

        self.score_file = os.path.join(score_dir, "scores.json")
        self.html_file = os.path.join(score_dir, "detailed_results.html")
        self.score_dir = score_dir
        logger.info(f"Read ingestion results from {prediction_dir}")
        html_heading("Detailed Results", self.html_file)

    def compute_scores(self, test_settings):
        """
        Compute the scores for the competition based on the test settings.

        Args:
            test_settings (dict): The test settings.
        """

        logger.info("Computing scores")

        # loop over ingestion results
        rmses, maes = [], []
        all_p16s, all_p84s, all_mus = [], [], []

        for i in range(len(self.ingestion_results)):
            ingestion_result = self.ingestion_results[i]
            mu = test_settings["ground_truth_mus"][i]

            mu_hats = ingestion_result["mu_hats"]
            delta_mu_hats = ingestion_result["delta_mu_hats"]
            p16s = ingestion_result["p16s"]
            p84s = ingestion_result["p84s"]

            all_mus.extend(np.repeat(mu, len(p16s)))
            all_p16s.extend(p16s)
            all_p84s.extend(p84s)

            set_rmses, set_maes = [], []
            for mu_hat, delta_mu_hat in zip(mu_hats, delta_mu_hats):
                set_rmses.append(self.RMSE_score(mu, mu_hat, delta_mu_hat))
                set_maes.append(self.MAE_score(mu, mu_hat, delta_mu_hat))
            set_interval, set_coverage, set_quantiles_score = self.Quantiles_Score(
                np.repeat(mu, len(p16s)), np.array(p16s), np.array(p84s)
            )

            set_mae = np.mean(set_maes)
            set_rmse = np.mean(set_rmses)

            result_text = f"Set {i} \nMAE: {set_mae.round(4)} \nRMSE: {set_rmse.round(4)} \nInterval: {set_interval.round(4)} \nCoverage: {set_coverage.round(4)} \nQuantiles Score: {set_quantiles_score.round(4)}"

            self.save_figure(
                mu=np.mean(mu_hats),
                p16s=p16s,
                p84s=p84s,
                set=i,
                true_mu=mu,
                result_text=result_text,
            )

            # Save set scores in lists
            rmses.append(set_rmse)
            maes.append(set_mae)

        overall_interval, overall_coverage, overall_quantiles_score = (
            self.Quantiles_Score(
                np.array(all_mus), np.array(all_p16s), np.array(all_p84s)
            )
        )

        self.scores_dict = {
            "rmse": round(np.mean(rmses), 4),
            "mae": round(np.mean(maes), 4),
            "interval": round(overall_interval, 4),
            "coverage": round(overall_coverage, 4),
            "quantiles_score": round(overall_quantiles_score, 4),
            "ingestion_duration": round(self.ingestion_duration, 4) if self.ingestion_duration is not None else None,
        }

        html_text(self.scores_dict, self.html_file, font_size="30px")
        print("[✔]")

    def RMSE_score(self, mu, mu_hat, delta_mu_hat):
        """
        Compute the root mean squared error between the true value mu and the predicted value mu_hat.

        Args:
            * mu (float): The true value.
            * mu_hat (np.array): The predicted value.
            * delta_mu_hat (np.array): The uncertainty on the predicted value.
        """

        def MSE(mu, mu_hat):
            """Compute the mean squared error between scalar mu and vector mu_hat."""
            return np.mean((mu_hat - mu) ** 2)

        def MSE2(mu, mu_hat, delta_mu_hat):
            """Compute the mean squared error between computed delta_mu = mu_hat - mu and delta_mu_hat."""
            adjusted_diffs = (mu_hat - mu) ** 2 - delta_mu_hat**2
            return np.mean(adjusted_diffs**2)

        return np.sqrt(MSE(mu, mu_hat) + MSE2(mu, mu_hat, delta_mu_hat))

    def MAE_score(self, mu, mu_hat, delta_mu_hat):
        """
        Compute the mean absolute error between the true value mu and the predicted value mu_hat.

        Args:
            * mu (float): The true value.
            * mu_hat (np.array): The predicted value.
            * delta_mu_hat (np.array): The uncertainty on the predicted value
        """

        def MAE(mu, mu_hat):
            """Compute the mean absolute error between scalar mu and vector mu_hat."""
            return np.mean(np.abs(mu_hat - mu))

        def MAE2(mu, mu_hat, delta_mu_hat):
            """Compute the mean absolute error based on the provided definitions."""
            adjusted_diffs = np.abs(mu_hat - mu) - delta_mu_hat
            return np.mean(np.abs(adjusted_diffs))

        return MAE(mu, mu_hat) + MAE2(mu, mu_hat, delta_mu_hat)

    def Quantiles_Score(self, mu, p16, p84, eps=1e-3):
        """
        Compute the quantiles score based on the true value mu and the quantiles p16 and p84.

        Args:
            * mu (array): The true ${\\mu} value.
            * p16 (array): The 16th percentile.
            * p84 (array): The 84th percentile.
            * eps (float, optional): A small value to avoid division by zero. Defaults to 1e-3.
        """

        def Interval(p16, p84):
            """Compute the average of the intervals defined by vectors p16 and p84."""
            interval = np.mean(p84 - p16)
            if interval < 0:
                logger.warning(f"Interval is negative: {interval}")
            return np.mean(abs(p84 - p16))

        def Coverage(mu, p16, p84):
            """Compute the fraction of times scalar mu is within intervals defined by vectors p16 and p84."""
            return_coverage = np.mean((mu >= p16) & (mu <= p84))
            return return_coverage

        def f(x, n_tries, max_coverage=1e4, one_sigma=0.6827):
            sigma68 = np.sqrt(((1 - one_sigma) * one_sigma * n_tries)) / n_tries

            if x >= one_sigma - 2 * sigma68 and x <= one_sigma + 2 * sigma68:
                out = 1
            elif x < one_sigma - 2 * sigma68:
                out = 1 + abs((x - (one_sigma - 2 * sigma68)) / sigma68) ** 4
            elif x > one_sigma + 2 * sigma68:
                out = 1 + abs((x - (one_sigma + 2 * sigma68)) / sigma68) ** 3
            return out

        coverage = Coverage(mu, p16, p84)
        interval = Interval(p16, p84)
        score = -np.log((interval + eps) * f(coverage, n_tries=mu.shape[0]))
        return interval, coverage, score

    def write_scores(self):

        logger.info(f"Writing scores to {self.score_file}")

        with open(self.score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))

    def save_figure(self, mu, p16s, p84s, set=0, true_mu=None, result_text=None):
        """
        Save the figure of the mu distribution.

        Args:
            * mu (array): The true ${\\mu} value.
            * p16 (array): The 16th percentile.
            * p84 (array): The 84th percentile.
            * set (int, optional): The set number. Defaults to 0.
        """
        
        plt.figure(figsize=(5, 4))
        # plot horizontal lines from p16 to p84
        for i, (p16, p84) in enumerate(zip(p16s, p84s)):
            if p16 > p84:
                p16, p84 = 0, 0
            if i == 0:
                plt.hlines(
                    y=i, xmin=p16, xmax=p84, colors="b", linewidth=2,label="Coverage interval"
                )
            else:
                plt.hlines(y=i, xmin=p16, xmax=p84, colors="b")
        plt.vlines(
            x=mu,
            ymin=0,
            ymax=len(p16s),
            colors="r",
            linewidth=2,
            linestyles="dashed",
            label="average $\\mu$",
        )
        if true_mu is not None:
            plt.vlines(
                x=true_mu,
                ymin=0,
                ymax=len(p16s),
                colors="g",
                linewidth=2,
                linestyles="dashed",
                label="true $\\mu$",
            )

        plt.xlabel("$\\mu$", fontdict={"size": 14})
        plt.ylabel("pseudo-experiments", fontdict={"size": 14})
        plt.xticks(fontsize=14)  # Set the x-tick font size
        plt.yticks(fontsize=14)  # Set the y-tick font size
        plt.xlim(min(p16s),max(p84s) + 1)
        plt.title(f"Set {set}", fontdict={"size": 14})
        plt.figtext(0.5, -0.3, result_text, wrap=True, horizontalalignment='center',
                    fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        plt.legend(loc="upper left", fontsize=12)
        plt.tight_layout()

        if result_text is None:
            result_text = f"Set {set} - $\\mu$ distribution"

        save_plot_to_html(plt, self.html_file, result_text, append=True)


def html_text(data, html_fle, font_size="20px"):
    
    bar_html = f"""
    <div class="table-box">
        <table>
            <tr><th>Key</th><th>Value</th></tr>
            {''.join(f'<tr><td>{key}</td><td>{value}</td></tr>' for key, value in data.items())}
        </table>
    </div>
    """
    with open(html_fle, "a") as f:
        f.write(bar_html)

def html_table(data,html_fle):
    # Build HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dictionary Table</title>
        <style>
            .table-box {{
                width: fit-content;
                padding: 20px;
                border: 2px solid #333;
                border-radius: 10px;
                background-color: #f9f9f9;
                margin: 40px auto;
                font-family: Arial, sans-serif;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            td, th {{
                border: 1px solid #666;
                padding: 10px 20px;
                text-align: left;
            }}
            th {{
                background-color: #ddd;
            }}
        </style>
    </head>
    <body>

    <div class="table-box">
        <table>
            <tr><th>Key</th><th>Value</th></tr>
            {''.join(f'<tr><td>{key}</td><td>{value}</td></tr>' for key, value in data.items())}
        </table>
    </div>

    </body>
    </html>
    """

    with open(html_fle, "a") as f:
        f.write(html_content)

def save_plot_to_html(plt, html_file, text, append=False):
    fig = plt.gcf()  # Get the current figure
    html_str = mpld3.fig_to_html(fig)
    formatted_text = text.replace("\n", "<br>")
    centered_html_str = f"""
    <div style="border: 2px solid black; padding: 10px;">
        <div style="display: flex; justify-content: center; align-items: center;">
            <div style="margin-right: 10px;">{html_str}</div>
            <div style="border: 1px solid black; padding: 10px; font-size: 20px;">{formatted_text}</div>
        </div>
    </div>
    """
    if append:
        with open(html_file, "a") as f:
            f.write(centered_html_str)
    else:
        bar_html = """
        <div style="background-color: lightgray; padding: 10px; text-align: center; font-size: 36px;">
            Detailed Results
        </div>
        """
        with open(html_file, "w") as f:
            f.write(centered_html_str)


def html_heading(heading, html_file):
    heading_html = f"""
    <div style="background-color: lightgray; padding: 10px; text-align: center; font-size: 20px;">
        {heading}
    </div>
    """
    with open(html_file, "w") as f:
        f.write(heading_html)
        
    


if __name__ == "__main__":
    print("############################################")
    print("### Scoring Program")
    print("############################################\n")

    import pathlib

    root_dir_name = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        description="This is script to generate data for the HEP competition."
    )
    parser.add_argument(
        "--prediction",
        "-p",
        type=pathlib.Path,
        help="Prediction file location",
        default=os.path.join(root_dir_name, "sample_result_submission"),
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file location",
        default=os.path.join(root_dir_name, "scoring_output"),
    )
    parser.add_argument(
        "--reference",
        "-r",
        help="Reference file location",
        default=os.path.join(root_dir_name, "reference_data"),
    )
    parser.add_argument(
        "--codabench",
        help="True when running on Codabench",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.codabench:
        prediction_dir = args.prediction
        output_dir = args.output
        reference_dir = args.reference
        program_dir = os.path.join(root_dir_name, "ingestion_program")
    else:
        prediction_dir = "/app/input/res"
        output_dir = "/app/output"
        reference_dir = "/app/input/ref"
        program_dir = os.path.join(root_dir_name, "ingestion_program")

    sys.path.append(program_dir)

    settings_file = os.path.join(prediction_dir, "test_settings.json")
    print(settings_file)
    try:
        with open(settings_file) as f:
            test_settings = json.load(f)
    except FileNotFoundError:
        settings_file = os.path.join(reference_dir, "settings", "data.json")
        try:
            with open(settings_file) as f:
                test_settings = json.load(f)
        except FileNotFoundError:
            print("Settings file not found. Please provide the settings file.")
            sys.exit(1)


    from HiggsML.score import Scoring


    # Init scoring
    scoring = Scoring()

    # Start timer
    scoring.start_timer()

    # Load ingestion duration
    ingestion_duration_file = os.path.join(prediction_dir, "ingestion_duration.json")
    scoring.load_ingestion_duration(ingestion_duration_file)

    print(prediction_dir)

    # Load ingestions results
    scoring.load_ingestion_results(prediction_dir, output_dir)

    # Compute Scores
    scoring.compute_scores(test_settings)

    # Write scores
    scoring.write_scores()

    # Stop timer
    scoring.stop_timer()

    print("\n----------------------------------------------")
    print("[✔] Scoring Program executed successfully!")
    print("----------------------------------------------\n\n")
