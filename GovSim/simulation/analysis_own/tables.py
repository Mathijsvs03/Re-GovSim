import json
import os
import argparse

def fill_latex_table_all_models(values):
    """
    Fill a LaTeX table template with values for a given model evaluation table.
    Args:
        values (list): A list of values to fill in the table.
                       The list must contain 38 elements (6 models x 6 columns per model).
    Returns:
        str: A filled LaTeX table as a string.
    """
    # Ensure the values list has the correct number of elements
    # if len(values) != 44:
    #     raise ValueError("The values list must contain exactly 38 elements (6 models x 6 metrics each).")

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
    R1-Distill-Llama-8b                                 & {24}                   & {25}                   & {26}                & {27}                & {28}              & {29}                \\
    R1-Distill-Qwen-14b                                 & {30}                   & {31}                   & {32}                & {33}                & {34}              & {35}                \\
    Phi-4                                               & {36}                   & {37}                   & {38}                & {39}                & {40}              & {41}                \\
    Qwen-14b                                            & {42}                   & {43}                   & {44}                & {45}                & {46}              & {47}                \\ \hline
    \end{{tabular}}
    \caption{{{48}}}
    \label{{tab:{49}}}
    \end{{table}}
    """

    # Format the template with the provided values
    filled_table = template.format(*values)
    return filled_table


def fill_table_subset_models(values):
    """
    Fill a LaTeX table template with values for a given model evaluation table.
    Args:
        values (list): A list of values to fill in the table.
                       The list must contain 38 elements (4 models x 6 columns per model).
    Returns:
        str: A filled LaTeX table as a string.
    """
    # Ensure the values list has the correct number of elements
    if len(values) != 26:
        raise ValueError("The values list must contain exactly 26 elements (2 models x 6 metrics each).")

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
    Mistral-7b                                          & {18}                   & {19}                   & {20}                & {21}                & {22}              & {23}                \\ \hline
    \end{{tabular}}
    \caption{{{24}}}
    \label{{tab:{25}}}
    \end{{table}}
    """

    # Format the template with the provided values
    filled_table = template.format(*values)
    return filled_table

def get_result_list(scenario, experiment, models):
    experiment = "universalization"
    path = f"results_json/{scenario}/{experiment}"

    if scenario == "fishing_v6.4":
        scen = "fish"
    elif scenario == "sheep_v6.4":
        scen = "sheep"
    else:
        scen  = "pollution"

    if experiment == "universalization":
        exp = "_universalization"
    elif experiment == "consequentialism":
        exp = "_consequentialism"
    elif experiment == "deontology":
        exp = "_deontology"
    elif experiment == "utilitarianism":
        exp = "_utilitarianism"
    elif experiment == "virtue_ethics":
        exp = "_virtue_ethics"
    elif experiment == "maximin_principle":
        exp = "_maximin_principle"
    elif experiment == "instruction":
        exp = "_instruction"
    elif experiment == "universalization_advice":
        exp = "_universalization_advice"
    else:
        exp = ""

    results = []
    for model in models:
        temp_results = {}
        model = f"{model}_{scen}_baseline_concurrent{exp}.json"

        if os.path.exists(os.path.join(path, model)) == False:
            # print(f"File not found: {os.path.join(path, model)}, exteding with zeros")
            results.extend(["0", "0", "0", "0", "0", "0"])
            continue

        with open(os.path.join(path, model), 'r') as file:
            temp_results = json.load(file)

            survival_rate = "{:.2f}".format(temp_results["general"]["survival_rate"]).replace("\\", "\\\\")

            survival_time = "{:.2f}".format(temp_results["general"]["mean_survival"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_survival'], 2))}}}"
            print(model)
            print(survival_time)
            gains = "{:.2f}".format(temp_results["general"]["mean_gains"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_gains'], 2))}}}"
            efficiency = "{:.2f}".format(temp_results["general"]["mean_efficiency"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_efficiency'], 2))}}}"
            equality = "{:.2f}".format(temp_results["general"]["mean_equality"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_equality'], 2))}}}"
            usage = "{:.2f}".format(temp_results["general"]["mean_over_usage"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_over_usage'], 2))}}}"
        print(results)
        results.extend([survival_rate, survival_time, gains, efficiency, equality, usage])

    if scenario == "fishing_v6.4":
        scenario = "Fishing"
    elif scenario == "sheep_v6.4":
        scenario = "Pasture"
    else:
        scenario = "Pollution"

    if experiment == "universalization":
        experiment = "Universalization"
    elif experiment == "consequentialism":
        experiment = "Consequentialism"
    elif experiment == "deontology":
        experiment = "Deontology"
    elif experiment == "utilitarianism":
        experiment = "Utilitarianism"
    elif experiment == "virtue_ethics":
        experiment = "Virtue Ethics"
    elif experiment == "maximin_principle":
        experiment = "Maximin Principle"
    elif experiment == "instruction":
        experiment = "Instruction"
    elif experiment == "universalization_advice":
        experiment = "Universalization Advice"
    else:
        experiment = "default"

    caption = f"Experiment: \\textit{{{experiment} - {scenario}}}"
    label = f"{scenario.lower()}_{experiment.lower()}"

    results.extend([caption, label])
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments with specified models.")
    parser.add_argument('models', type=str, nargs='?', default=None, help="Comma-separated list of models to run experiments with")
    parser.add_argument('experiments', type=str, nargs='?', default=None, help="Comma-separated list of experiments to run")
    parser.add_argument('--all', action='store_true', help="Select all models and experiments (default behavior if not specified)")

    args = parser.parse_args()

    all_models = [
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Meta-Llama-3-8B-Instruct",
        "Mistral-7B-Instruct-v0.2",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-14B",
        "phi-4",
        "Qwen-14B"
    ]
    all_experiments = ["baseline", "universalization", "instruction", "universalization_advice", "consequentialism", "deontology", "maximin_principle", "universalization", "utilitarianism", "virtue_ethics"]

    if args.all:
        models = all_models
        experiments = ["baseline"]
    else:
        models = all_models if args.models is None else args.models.split(',')
        experiments = all_experiments


    scenarios = [
        "fishing_v6.4",
        "sheep_v6.4",
        "pollution_v6.4"
    ]
    if any(model in models for model in ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-14B", "phi-4"]):
        for scenario in scenarios:
            for experiment in experiments:
                # Get the results for the specified scenario and experiment
                results = get_result_list(scenario, experiment, models)
                table = fill_latex_table_all_models(results)
                print(table)

                with open(f"tables/{scenario}_{experiment}.tex", "w") as file:
                    file.write(table)
    else:
        for scenario in scenarios:
            for experiment in experiments:
                # Get the results for the specified scenario and experiment
                results = get_result_list(scenario, experiment, models)
                table = fill_table_subset_models(results)
                print(table)

                with open(f"tables/{scenario}_{experiment}.tex", "w") as file:
                    file.write(table)
