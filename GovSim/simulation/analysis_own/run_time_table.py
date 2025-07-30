"""
This script processes runtime data from various models and scenarios, calculates the mean runtimes,
and generates a LaTeX table to display the results. The script reads JSON files containing runtime
data, computes the average runtime for each model and scenario, and formats the results into a LaTeX
table for easy inclusion in reports or publications.

Functions:
- fill_latex_table(values): Fills a LaTeX table template with provided values.
- get_result_list(runtimes): Computes the mean runtime for each model and scenario from the runtimes dictionary.

Usage:
Run the script directly to process the runtime data and print the LaTeX table.
"""

import os
import json
import numpy as np

def fill_latex_table(values):
    """
    Fill a LaTeX table template with values for a given model evaluation table.
    Args:
        values (list): A list of values to fill in the table.
                       The list must contain 18 elements (4 models x 3 metrics each).
                       The list must contain 18 elements (4 models x 3 metrics each).
    Returns:
        str: A filled LaTeX table as a string.
    """
    # Ensure the values list has the correct number of elements
    if len(values) != 18:
        raise ValueError("The values list must contain exactly 18 elements (4 models x 3 metrics each).")
    if len(values) != 18:
        raise ValueError("The values list must contain exactly 18 elements (4 models x 3 metrics each).")

    # Define the LaTeX table template with placeholders
    template = r"""
    \begin{{table}}[]
    \begin{{tabular}}{{lccc}}
                    & Fishing     & Pasture    & Pollution    \\ \hline
    \multicolumn{{4}}{{l}}{{\textit{{\textbf{{Open-Weights Models}}}}}} \\
    Llama-2-7b      & {0}         & {1}        & {2}          \\
    Llama-2-13b     & {3}         & {4}        & {5}          \\
    Llama-3-8b      & {6}         & {7}        & {8}          \\
    Mistral-7b      & {9}         & {10}       & {11}         \\
    R1-Llama-8B     & {12}        & {13}       & {14}         \\
    R1-Qwen-14B     & {15}        & {16}       & {17}         \\ \hline
    \end{{tabular}}
    \end{{table}}
    """

    # Format the template with the provided values
    filled_table = template.format(*values)
    return filled_table

def get_result_list(runtimes):
    results = []

    for model in runtimes:
        for scenario in runtimes[model]:
            results.append(np.mean(runtimes[model][scenario]))

    return results

if __name__ == '__main__':
    scenario = "fishing_v6.4"
    experiment = "baseline"

    scenarios = ["fishing_v6.4", "sheep_v6.4", "pollution_v6.4"]
    experiments = ["baseline", "universalization"]

    runtimes = {}

    for scenario in scenarios:
        for experiment in experiments:
            path = f"results_json/{scenario}/{experiment}"
            models = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Meta-Llama-3-8B-Instruct", "Mistral-7b-Instruct-v0.2", \
                        "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-14B"]

            if scenario == "fishing_v6.4":
                scen = "fish"
            elif scenario == "sheep_v6.4":
                scen = "sheep"
            else:
                scen  = "pollution"

            if experiment == "universalization":
                exp = "_universalization"
            else:
                exp = ""


            for model in models:
                if (model == "DeepSeek-R1-Distill-Llama-8B" or model == "DeepSeek-R1-Distill-Qwen-14B") and experiment == "universalization":
                    continue

                if (model == "DeepSeek-R1-Distill-Llama-8B" or model == "DeepSeek-R1-Distill-Qwen-14B") and experiment == "universalization":
                    continue

                model_long = f"{model}_{scen}_baseline_concurrent{exp}.json"

                temp_runtimes = []
                with open(os.path.join(path, model_long), 'r') as file:
                    temp_results = json.load(file)
                    temp_results.pop("general")

                    for run in temp_results:
                        print(temp_results[run])
                        temp_runtimes.append(temp_results[run]["runtime"])



                if model not in runtimes:
                    runtimes[model] = {}

                    if scenario not in runtimes[model]:
                        runtimes[model][scenario] = []

                    runtimes[model][scenario].extend(temp_runtimes)

                else:
                    if scenario not in runtimes[model]:
                        runtimes[model][scenario] = []

                    runtimes[model][scenario].extend(temp_runtimes)

    results = get_result_list(runtimes)
    print(fill_latex_table(results))