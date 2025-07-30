"""
This script aggregates model evaluation metrics across multiple scenarios and experiments,
and generates a LaTeX table summarizing the results.

Functions:
    fill_latex_table(values):

Main Execution:
    - Aggregates results over specified scenarios and experiments.
    - Computes average metrics and confidence intervals for each model.
    - Generates a LaTeX table with the aggregated results.
    - Saves the LaTeX table to a file.
"""


import json
import os
import numpy as np
from metrics import calculate_confidence_interval

def fill_latex_table(values):
    """
    Fill a LaTeX table template with values for a given model evaluation table.
    Args:
        values (list): A list of values to fill in the table.
                       The list must contain 38 elements
    Returns:
        str: A filled LaTeX table as a string.
    """
    # Ensure the values list has the correct number of elements
    if len(values) != 38:
        raise ValueError("The values list must contain exactly 38 elements.")

    # Define the LaTeX table template with placeholders
    template = r"""
    \begin{{table}}[H]
    \centering
    \setlength\tabcolsep{{5pt}} % default value: 6pt
    \begin{{tabular}}{{lcccccc}}
    \hline
    \multicolumn{{1}}{{c}}{{\multirow{{2}}{{*}}{{\textbf{{Model}}}}}} & \textbf{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}Survival\\ Rate\end{{tabular}}}} & \textbf{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}Survival\\ Time\end{{tabular}}}} & \textbf{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}Total\\ Gain\end{{tabular}}}} & \textbf{{Efficiency}} & \textbf{{Equality}} & \textbf{{Over-usage}} \\
    \multicolumn{{1}}{{c}}{{}}                                & Max = 100              & Max = 12               & Max = 120           & Max = 100           & Max = 1           & Min = 0             \\ \hline
    \multicolumn{{7}}{{l}}{{\textit{{\textbf{{Open-Weights Models}}}}}}                                                                                                                                   \\
    Llama-2-7b                                          & {0}                    & {1}                    & {2}                 & {3}                 & {4}               & {5}                 \\
    Llama-2-13b                                         & {6}                    & {7}                    & {8}                 & {9}                 & {10}              & {11}                \\
    Llama-3-8b                                          & {12}                   & {13}                   & {14}                & {15}                & {16}              & {17}                \\
    Mistral-7b                                          & {18}                   & {19}                   & {20}                & {21}                & {22}              & {23}                \\
    DeepSeek-R1-Llama-8b                                & {24}                   & {25}                   & {26}                & {27}                & {28}              & {29}                \\
    DeepSeek-R1-Qwen-14B                                & {30}                   & {31}                   & {32}                & {33}                & {34}              & {35}                \\ \hline
    \end{{tabular}}
    \caption{{{36}}}
    \label{{tab:{37}}}
    \end{{table}}
    """

    # Format the template with the provided values
    filled_table = template.format(*values)
    return filled_table


if __name__ == '__main__':
    # We will aggregate the results over these scenarios and the single experiment "baseline".
    scenarios = [["fishing_v6.4", "fish"], ["sheep_v6.4", "sheep"], ["pollution_v6.4", "pollution"]]
    experiments = ["baseline"]
    test = "baseline_concurrent"
    models = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2", \
            "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-14B"]

    # Dictionary to gather all metrics across scenarios for each model
    aggregator = {
        m: {
            "survival_rate": [],
            "survival_time": [],
            "gains": [],
            "efficiency": [],
            "equality": [],
            "over_usage": []
        }
        for m in models
    }

    base_path = "../results/"

    # Collect data from all scenarios (and experiments)
    for scenario, task in scenarios:
        for experiment in experiments:
            for model in models:
                path = f"results_json/{scenario}/{experiment}"
                group = f"{model}_{task}_{test}"
                path_runs = os.path.join(base_path, scenario, group)
                runs = os.listdir(path_runs)

                with open(f"{os.path.join(path, group)}.json", 'r') as file:
                    data = json.load(file)

                for run in runs:
                    aggregator[model]["survival_time"].append(data[run]["survival_time"])
                    aggregator[model]["gains"].append(np.mean(list(data[run]["gains"].values())))
                    aggregator[model]["efficiency"].append(data[run]["efficiency"])
                    aggregator[model]["equality"].append(data[run]["equality"])
                    aggregator[model]["over_usage"].append(data[run]["over_usage"])

                aggregator[model]["survival_rate"].append(data["general"]["survival_rate"])

    # Now compute the average for each model across the 3 scenarios
    # We'll produce 6 columns per model (each as a formatted string).
    results_by_model = []

    for model, dict in aggregator.items():
        mean_survival_rate = np.mean(dict["survival_rate"])
        mean_survival_time = np.mean(dict["survival_time"])
        moe_confidence_survival = calculate_confidence_interval(dict["survival_time"])[3]
        mean_gain = np.mean(dict["gains"])
        moe_confidence_gain =  calculate_confidence_interval(dict["gains"])[3]
        mean_efficiency = np.mean(dict["efficiency"])
        moe_confidence_efficiency =  calculate_confidence_interval(dict["efficiency"])[3]
        mean_equality = np.mean(dict["equality"])
        moe_confidence_equality =  calculate_confidence_interval(dict["equality"])[3]
        mean_over_usage = np.mean(dict["over_usage"])
        moe_confidence_overusage =  calculate_confidence_interval(dict["over_usage"])[3]
        # Convert each metric to a string with 2 decimals
        # (Adjust formatting as needed, e.g., ± confidence intervals, etc.)
        results_by_model.extend([
            f"{mean_survival_rate:.2f}".replace("\\", "\\\\"),
            f"{mean_survival_time:.2f}".replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(moe_confidence_survival, 2))}}}",
            f"{mean_gain:.2f}".replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(moe_confidence_gain, 2))}}}",
            f"{mean_efficiency:.2f}".replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(moe_confidence_efficiency, 2))}}}",
            f"{mean_equality:.2f}".replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(moe_confidence_equality, 2))}}}",
            f"{mean_over_usage:.2f}".replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(moe_confidence_overusage, 2))}}}"
        ])

    # Add caption and label for the aggregated table
    caption = "Aggregated baseline results (3 scenarios)"
    label = "aggregated_baseline"

    # Final list of 4×6 + 2 = 26 elements
    results_by_model.extend([caption, label])

    filled_table = fill_latex_table(results_by_model)
    print(filled_table)

    with open(f"tables/extended_aggregated_scenarios.tex", "w") as file:
        file.write(filled_table)