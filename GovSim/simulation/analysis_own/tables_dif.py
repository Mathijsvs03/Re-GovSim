import json
import os

def fill_latex_table(values):
    """
    Fill a LaTeX table template with values for a given model evaluation table.
    Args:
        values (list): A list of values to fill in the table.
                       The list must contain 24 elements (4 models x 6 columns per model).
    Returns:
        str: A filled LaTeX table as a string.
    """
    # Ensure the values list has the correct number of elements
    # if len(values) != 26:
    #     raise ValueError("The values list must contain exactly 24 elements (4 models x 6 metrics each).")

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
    \caption{{{24}}}
    \label{{tab:{25}}}
    \end{{table}}
    """

    # Format the template with the provided values
    print(values)
    filled_table = template.format(*values)
    return filled_table

def experiment_to_experiment_route(experiment):
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
    else:
        exp = ""

    return exp

def format_result_pos_neg(diff, over_usage=False):
    diff = round(diff, 2)
    if diff == 0:
        return f"{diff}"

    if over_usage:
        if diff > 0:
            return f"\\textcolor{{red}}{{{diff}$\\shortuparrow$}}"
        else:
            return f"\\textcolor{{ForestGreen}}{{+{diff}$\\shortdownarrow$}}"
    else:
        if diff > 0:
            return f"\\textcolor{{ForestGreen}}{{+{diff}$\\shortuparrow$}}"
        else:
            return f"\\textcolor{{red}}{{{diff}$\\shortdownarrow$}}"

def experiment_to_caption_label(experiment):
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
    else:
        experiment = "default"

    return experiment


def get_result_list_compare(scenario, experiments):
    experiment_0, experiment_1 = experiments
    path1 = f"results_json/{scenario}/{experiment_0}"
    path2 = f"results_json/{scenario}/{experiment_1}"

    models = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2", "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-14B", "phi-4", "Qwen-14B"]
    if scenario == "fishing_v6.4":
        scen = "fish"
    elif scenario == "sheep_v6.4":
        scen = "sheep"
    else:
        scen  = "pollution"

    exp0 = experiment_to_experiment_route(experiment_0)
    exp1 = experiment_to_experiment_route(experiment_1)

    results = []
    for model in models:
        comp_results = {}
        model0 = f"{model}_{scen}_baseline_concurrent{exp0}.json"
        model1 = f"{model}_{scen}_baseline_concurrent{exp1}.json"
        with open(os.path.join(path1, model0), 'r') as file:
            temp_results = json.load(file)
            survival_rate0 = (temp_results["general"]["survival_rate"])
            survival_time0 = (temp_results["general"]["mean_survival"])
            gains0 = (temp_results["general"]["mean_gains"])
            efficiency0 = (temp_results["general"]["mean_efficiency"])
            equality0 = (temp_results["general"]["mean_equality"])
            usage0 = (temp_results["general"]["mean_over_usage"])

        if os.path.exists(os.path.join(path2, model1)) == False:
            # print(f"File not found: {os.path.join(path2, model1)}")
            results.extend(["0", "0", "0", "0", "0", "0"])
            continue

        with open(os.path.join(path2, model1), 'r') as file:
            temp_results = json.load(file)
            survival_rate1 = (temp_results["general"]["survival_rate"])
            survival_time1 = (temp_results["general"]["mean_survival"])
            gains1 = (temp_results["general"]["mean_gains"])
            efficiency1 = (temp_results["general"]["mean_efficiency"])
            equality1 = (temp_results["general"]["mean_equality"])
            usage1 = (temp_results["general"]["mean_over_usage"])

        diff_survival_rate = format_result_pos_neg(survival_rate1 - survival_rate0)
        diff_survival_time = format_result_pos_neg(survival_time1 - survival_time0)
        diff_gains = format_result_pos_neg(gains1 - gains0)
        diff_efficiency = format_result_pos_neg(efficiency1 - efficiency0)
        diff_equality = format_result_pos_neg(equality1 - equality0)
        diff_usage = format_result_pos_neg(usage1 - usage0)

        results.extend([diff_survival_rate, diff_survival_time, diff_gains, diff_efficiency, diff_equality, diff_usage])

    if scenario == "fishing_v6.4":
        scenario = "Fishing"
    elif scenario == "sheep_v6.4":
        scenario = "Pasture"
    else:
        scenario = "Pollution"

    exp_label_0 = experiment_to_caption_label(experiment_0)
    exp_label_1 = experiment_to_caption_label(experiment_1)

    caption = f"Comparison between: \\textit{{{exp_label_0} \\text{{and}} {exp_label_1} - {scenario}}}"
    label = f"{scenario.lower()}_{experiment.lower()}"

    results.extend([caption, label])
    return results

if __name__ == '__main__':
    scenario = "fishing_v6.4"
    experiment = "baseline"

    scenarios = [
        "fishing_v6.4",
        "sheep_v6.4",
        # "pollution_v6.4"
    ]
    experiments = ["baseline", "universalization", "instruction", "universalization_advice", "consequentialism", "deontology", "maximin_principle", "universalization", "utilitarianism", "virtue_ethics"]

    for scenario in scenarios:
        results = get_result_list_compare(scenario, ("baseline", "universalization"))
        table = fill_latex_table(results)

        with open(f"tables/{scenario}_conpare_base-base_advice.tex", "w") as file:
            file.write(table)