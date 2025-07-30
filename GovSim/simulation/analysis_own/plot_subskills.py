"""
This script processes simulation and subskill results, performs statistical analysis,
and generates plots to visualize the relationship between test case scores and survival
times across multiple groups and models. It provides functions for extracting data from
JSON files, performing linear regression and t-tests, and plotting the results for
subskills and aggregated scores.

The key functionalities of this script include:
- Extracting survival times and test case results from JSON files.
- Performing Ordinary Least Squares (OLS) linear regression to calculate R² values.
- Conducting t-tests to assess statistical significance between two datasets.
- Plotting individual subskills and their relationships with survival time.
- Plotting aggregated results across different experiment types and models.

Command-line Arguments:
- --simulation_path: Path to the directory containing simulation results (default: 'results_json').
- --subskill_path: Path to the directory containing subskill results (default: '../../subskills/results').
- --simulation_type: Type of simulation (choices: 'baseline', 'universalization', default: 'baseline').
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import warnings
import pandas as pd
import seaborn as sns


def extract_subdirectory_names(path: str):
    """
    Function to extract the subdirectory names from a path.

    Args:
        path (str): Path to the directory.

    Returns:
        list: A list of subdirectory names.
    """
    subdirectory_names = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            subdirectory_names.append(dir_name)
        break # only get the first level directories
    return subdirectory_names


def extract_aggregated_results_metric(path: str, metric: str):
    """
    Function to extract survival times from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        list: A list of survival times extracted from the JSON file.
    """
    mapping = {"Mean survival time": "mean_survival",
               "Survival rate": "survival_rate",
               "Mean gain": "mean_gains",
               "Mean efficiency": "mean_efficiency",
               "Mean equality": "mean_equality",
               "Mean overusage": "mean_over_usage"}

    with open(path, 'r') as f:
        data = json.load(f)

        for entrance, values in data.items():
            if entrance == "general":
                result_metric = values[mapping[metric]]

    return result_metric

def extract_test_case_results(path: str):
    """
    Function to extract the test case results from the test json file

    Args:
        path: path to the test Json file.

    Returns:

    """

    mean_test_score = 0
    std_test_score = 0

    with open(path, 'r') as f:
        data = json.load(f)

        for entrance, values in data.items():
            if entrance == "score_mean":
                mean_test_score = values
            elif entrance == "score_std":
                std_test_score = values

    return mean_test_score, std_test_score


def plot_subskills(x, y, labels, title, sub_titles, x_label, y_label, simulation_type):
    """
    Function to plot the subskills results in a single row, with markers showing LLM performance.

    Args:
        x (dict): A dictionary containing the x values (test scores) for each subskill.
        y (dict): A dictionary containing the y values (survival times) for each subskill.
        labels (iterable): A list of labels for each LLM (e.g., model names).
        title (str): The title of the overall plot.
        sub_titles (list): A list of sub-titles for each subskill (test case names).
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        simulation_type (str): The type of simulation (baseline or universalization).
    """
    plt.rcParams.update({
    "font.size": 14})
    plt.subplots_adjust(wspace=0.4)
    num_subskills = len(x)
    cols = num_subskills
    rows = 1

    mapping_y_lim = {"Mean survival time": 12,
                "Survival rate": 100,
                "Mean gain": 100,
                "Mean efficiency": 100,
                "Mean equality": 100,
                "Mean overusage": 120}

    mapping_y_lim_min = {"Mean survival time": 0,
                "Survival rate": 0,
                "Mean gain": 0,
                "Mean efficiency": 0,
                "Mean equality": 0,
                "Mean overusage": 0}


    if num_subskills > 1:
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 5.5 * rows))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))


    if num_subskills != 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    markers = ['X', 's', 'D', '^', '*', 'o']
    marker_sizes = [100, 100, 100, 100, 100, 100]
    x_labels = ["Resource estimation accuracy", "Threshold accuracy (assuming equal use)", "Inferred threshold accuracy", "Proportion sustainable action"]
    sub_skill_names = ["Simulation dynamics", "Sustainability threshold (assumption)", "Sustainability threshold (belief)", "Sustainable action"]
    for idx, (subskill_name, x_values) in enumerate(sorted(x.items(), key=lambda item: item[0])):
        ax = axes[idx]

        y_values = y[subskill_name]

        for i, (x_val, y_val) in enumerate(zip(x_values, y_values)):
            marker = markers[i % len(markers)]
            size = marker_sizes[i % len(marker_sizes)]
            ax.scatter(x_val, y_val, label=labels[i], marker=marker, s=size)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(mapping_y_lim_min[y_label], mapping_y_lim[y_label])
        ax.set_title(f'{sub_skill_names[idx]}', fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel(x_labels[idx], fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(True)

    for ax in axes[num_subskills:]:
        ax.axis("off")

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:

        if num_subskills > 1:
            fig.legend(handles, labels, loc='upper center', ncol=int(len(labels)/2), fontsize='x-large', bbox_to_anchor=(0.5, 1.0))
        else:
            fig.legend(handles, labels, loc='upper center', ncol=1, fontsize='x-large', bbox_to_anchor=(0.5, 0.92))

    # if simulation_type == 'baseline':
    #     fig.suptitle(f'{title}', fontsize=14, fontweight='bold')
    # elif simulation_type == 'universalization':
    #     fig.suptitle(title + ' (Universalization)', fontsize=14, fontweight='bold')

    if num_subskills > 1:
        plt.tight_layout(rect=[0, 0, 1, 0.8])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.85])

    plot_dir = "figs"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_path = os.path.join(plot_dir, f"{title.replace(' ', '_')+ '_' + simulation_type + '_' + '_'.join(y_label.split())}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_subskills_scenarios(x, y, labels, title, sub_titles, x_label, y_label, simulation_type):
    """
    Function to plot the subskills results in a single row, with markers showing LLM performance.

    Args:
        x (dict): A dictionary containing the x values (test scores) for each subskill.
        y (dict): A dictionary containing the y values (survival times) for each subskill.
        labels (iterable): A list of labels for each LLM (e.g., model names).
        title (str): The title of the overall plot.
        sub_titles (list): A list of sub-titles for each subskill (test case names).
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        simulation_type (str): The type of simulation (baseline or universalization).
    """
    plt.rcParams.update({
    "font.size": 14})
    plt.subplots_adjust(wspace=0.4)
    num_subskills = len(x)

    cols = num_subskills
    rows = 1

    mapping_y_lim = {"Mean survival time": 12,
                    "Survival rate": 100,
                    "Mean gain": 100,
                    "Mean efficiency": 100,
                    "Mean equality": 100,
                    "Mean overusage": 120}

    mapping_y_lim_min = {"Mean survival time": 0,
                "Survival rate": 0,
                "Mean gain": 0,
                "Mean efficiency": 0,
                "Mean equality": 0,
                "Mean overusage": 0}


    if num_subskills > 1:
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 5.5 * rows))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))


    if num_subskills != 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # fishing, pollution, sheep
    markers = ['^', 's', 'o']
    marker_sizes = [100, 100, 100]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'brown', 'purple']
    x_labels = ["Resource estimation accuracy", "Threshold accuracy (assuming equal use)", "Inferred threshold accuracy", "Proportion sustainable action"]

    for idx, (subskill_name, results_dict) in enumerate(sorted(x.items(), key=lambda item: item[0])):
        for jdx, (model_name, x_arr) in enumerate(sorted(results_dict.items(), key=lambda item: item[0])):
            ax = axes[idx]

            y_values = y[subskill_name][model_name]

            for i, (x_val, y_val) in enumerate(zip(x_arr, y_values)):
                size = marker_sizes[i % len(marker_sizes)]
                ax.scatter(x_val, y_val, marker=markers[i], s=size, c=colors[jdx])

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(mapping_y_lim_min[y_label], mapping_y_lim[y_label])
        ax.set_title(f'{subskill_name}', fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel(x_labels[idx], fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(True)

    for ax in axes[num_subskills:]:
        ax.axis("off")

    model_names = sorted(list(set(model_name for subdict in x.values() for model_name in subdict)))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=model_name,
            markerfacecolor=colors[i % len(colors)], markersize=10)
        for i, model_name in enumerate(model_names)
    ]

    # Plot legend
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=len(legend_elements) // 2 if len(legend_elements) > 2 else 1,
        fontsize='x-large',
        bbox_to_anchor=(0.5, 0.92)
    )

    # if simulation_type == 'baseline':
    #     fig.suptitle(f'{title}', fontsize=14, fontweight='bold')
    # elif simulation_type == 'universalization':
    #     fig.suptitle(title + ' (Universalization)', fontsize=14, fontweight='bold')

    if num_subskills > 1:
        plt.tight_layout(rect=[0, 0, 1, 0.7])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.85])

    plot_dir = "figs"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_path = os.path.join(plot_dir, f"{'scenarios' + '_' + '_'.join(y_label.split()) + '_' + simulation_type}.png")
    plt.savefig(plot_path)
    plt.close()

def plot_correlation_per_scenario(data, group_name, simulation_type, subskill_cols, metric_cols):

    df = pd.DataFrame(data)
    subskills = df[subskill_cols]

    corr_matrix = pd.DataFrame(index=subskills.columns, columns=df.columns)

    for sub in subskills.columns:
        for data in df.columns:
            corr_matrix.loc[sub, data] = subskills[sub].corr(df[data])
    corr_matrix = corr_matrix.astype(float)
    # corr = df.corr(method="pearson")
    plt.figure(figsize=(13, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    # plt.title(f"Correlation Between Subskills and Metrics scenario {group_name} and simulation type {simulation_type}")

    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot_kws={"size": 12},         # Font size for annotations in cells
        cbar_kws={"label": "Correlation", "shrink": 0.9}  # Add label to colorbar
    )

    # Increase font size for axis tick labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), size=13, rotation=45, ha="right")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), size=13, rotation=0)


    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Correlation", fontsize=0)
    plt.tight_layout()

    plot_dir = "figs"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_path = os.path.join(plot_dir, f"{'correlation_' + simulation_type + '_' + group_name}.png")
    plt.savefig(plot_path)
    plt.close()




def calculate_OLS_linear_regression(x, y):
    """
    Function to calculate the OLS linear regression for the given data.

    Args:
        x (list): A list of x values.
        y (list): A list of y values.

    Returns:
        float: R² value of the linear regression.
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    return r_sq


def t_test(x, y):
    """
    Function to perform a t-test on the given data.

    Args:
        x (list): A list of x values.
        y (list): A list of y values.

    Returns:
        float: T-test value.
    """

    # Supress warnings because of the small sample size and same value for
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = ttest_ind(x, y)

    return t_stat, p_value


def main():
    parser = argparse.ArgumentParser(description="Process simulation and subskill paths.")
    parser.add_argument('--simulation_path', type=str, help='Path to the simulation results JSON files.', default='results_json')
    parser.add_argument('--subskill_path', type=str, help='Path to the subskill results.', default='../../subskills/results')
    parser.add_argument('--simulation_type', type=str, choices=['baseline', 'universalization'], help='Type of simulation for determining which survival time is used', default='baseline')

    args = parser.parse_args()
    simulation_path = args.simulation_path
    subskill_path = args.subskill_path
    simulation_type = args.simulation_type

    group_names = extract_subdirectory_names(simulation_path)

    # dictionary to store the model name and the survival time and has
    # the structure {group_name: {model_name: {experiment_type: survival_time_list}}}
    model_metric_results = {}
    metrics = ["Mean survival time", "Survival rate", "Mean gain", "Mean efficiency", "Mean equality", "Mean overusage"]

    # fill the dictionary
    for metric in metrics:
        model_metric_results[metric] = {}

    for group_name in group_names:
        for metric in metrics:

            experiment_types = extract_subdirectory_names(os.path.join(simulation_path, group_name))
            model_metric_results[metric][group_name.split('_')[0]] = {}
            for experiment_type in experiment_types:
                # filtering
                if 'baseline' == experiment_type or 'universalization' == experiment_type:
                    new_path = os.path.join(simulation_path, group_name, experiment_type)
                    for root, dirs, files in os.walk(new_path):
                        for file in files:
                            if file.endswith('.json'):
                                file_path = os.path.join(new_path, file)
                                model_name = file.split('_')[0]
                                if model_name not in model_metric_results[metric][group_name.split('_')[0]]:
                                    model_metric_results[metric][group_name.split('_')[0]][model_name] = {}
                                if experiment_type not in model_metric_results[metric][group_name.split('_')[0]][model_name]:
                                    model_metric_results[metric][group_name.split('_')[0]][model_name][experiment_type] = {}
                                result_metric = extract_aggregated_results_metric(file_path, metric)
                                model_metric_results[metric][group_name.split('_')[0]][model_name][experiment_type] = result_metric

    test_file_2_test_case_baseline = {
        'multiple_sim_consequence_after_fishing_same_amount.json': 'simulation dynamics',
        'multiple_sim_catch_fish_standard_persona.json' : 'sustainable action',
        'multiple_sim_shrinking_limit_assumption.json': 'sustainability threshold (assumption)',
        'multiple_sim_shrinking_limit.json' : 'sustainability threshold (belief)',
        'multiple_sim_consequence_after_using_same_amount.json': 'simulation dynamics',
        'multiple_sim_consume_grass_standard_persona.json' : 'sustainable action',
    }

    test_file_2_test_case_univerlisation = {
        'multiple_sim_universalization_catch_fish.json': 'sustainable action',
        'multiple_sim_universalization_consume_grass.json' : 'sustainable action',
        'multiple_sim_universalization_consequence_after_fishing_same_amount.json': 'simulation dynamics',
        'multiple_sim_universalization_shrinking_limit_assumption.json': 'sustainability threshold (assumption)',
        'multiple_sim_universalization_shrinking_limit.json' : 'sustainability threshold (belief)',
        'multiple_sim_universalization_consequence_after_using_same_amount.json': 'simulation dynamics', }

    # dictionary to store the model name and the survival time and has
    # the structure {group_name: {model_name {test_name: mean_test_score}}}
    model_test_case = {}

    model_names = extract_subdirectory_names(subskill_path)

    for model_name in sorted(model_names):

        group_name = model_name.split('_')[1]
        if group_name not in model_test_case:
            model_test_case[group_name] = {}
        model_test_case[group_name][model_name.split('_')[0]] = {}

        run_names = extract_subdirectory_names(os.path.join(subskill_path, model_name))

        # Calculate the average test score for each model and specific test case
        test_name_mean_score = {}
        for run_name in run_names:
            run_path = os.path.join(subskill_path, model_name, run_name)
            test_names = []
            for root, dirs, files in os.walk(run_path):
                for file in files:
                    if file.endswith('.json'):
                        test_names.append(file)

            for test_name in test_names:
                if simulation_type == 'baseline':
                    if test_name not in test_file_2_test_case_baseline:
                        continue
                    test_case_name = test_file_2_test_case_baseline[test_name]
                    test_case_path = os.path.join(run_path, test_name)
                    mean_test_score, _ = extract_test_case_results(test_case_path)
                    if test_case_name not in test_name_mean_score:
                        test_name_mean_score[test_case_name] = []
                    test_name_mean_score[test_case_name].append(mean_test_score)
                elif simulation_type == 'universalization':
                    if test_name not in test_file_2_test_case_univerlisation:
                        continue
                    test_case_name = test_file_2_test_case_univerlisation[test_name]
                    test_case_path = os.path.join(run_path, test_name)
                    mean_test_score, _ = extract_test_case_results(test_case_path)
                    if test_case_name not in test_name_mean_score:
                        test_name_mean_score[test_case_name] = []
                    test_name_mean_score[test_case_name].append(mean_test_score)
        for test_name, mean_scores in test_name_mean_score.items():

            model_test_case[group_name][model_name.split('_')[0]][test_name] = np.mean(mean_scores)

    # Plot the individual subskills
    # for metric in metrics:
    #     for group_name, model_dict in model_test_case.items():
    #         x = {}
    #         y = {}
    #         sub_titles = []
    #         labels = []
    #         for model_name, test_case_dict in model_dict.items():

    #             labels.append(model_name)
    #             for test_case_name, test_case_score in test_case_dict.items():
    #                 if test_case_name not in x:
    #                     x[test_case_name] = []
    #                     y[test_case_name] = []
    #                 y[test_case_name].append(model_metric_results[metric][group_name][model_name][simulation_type])
    #                 x[test_case_name].append(test_case_score)

    #                 if test_case_name not in sub_titles:
    #                     sub_titles.append(test_case_name)

    #         for subskill_name, x_values in x.items():
    #             y_values = y[subskill_name]
    #             r_sq = calculate_OLS_linear_regression(x_values, y_values)
    #             t_test_val, p_val = t_test(x_values, y_values)
    #             print(f"R² value for group name {group_name} and subskill name {subskill_name}: {r_sq} with p_value: {p_val}")

    #         plot_subskills(x=x, y=y, labels=labels, title=f"Subskills for {group_name}", sub_titles=sub_titles, x_label="Test Case Score", y_label=metric, simulation_type=simulation_type)

    # # Plot the subskills averaged across all three groups (experiment types) per LLM
    # all_x = {}
    # all_y = {}
    # all_sub_titles = []

    # test_file_mapping = (
    #     test_file_2_test_case_baseline if simulation_type == "baseline" else test_file_2_test_case_univerlisation
    # )
    # for metric in metrics:
    #     for group_name, model_dict in model_test_case.items():

    #         for model_name, test_case_dict in model_dict.items():


    #             for test_file_name, test_case_name in test_file_mapping.items():
    #                 if test_case_name not in test_case_dict:
    #                     continue

    #                 if model_name not in all_x:
    #                     all_x[model_name] = {}
    #                     all_y[model_name] = {}

    #                 if test_case_name not in all_x[model_name]:
    #                     all_x[model_name][test_case_name] = []
    #                     all_y[model_name][test_case_name] = []


    #                 test_case_score = test_case_dict[test_case_name]
    #                 survival_time = model_metric_results[metric][group_name][model_name][simulation_type]

    #                 all_x[model_name][test_case_name].append(test_case_score)
    #                 all_y[model_name][test_case_name].append(survival_time)

    #                 if test_case_name not in all_sub_titles:
    #                     all_sub_titles.append(test_case_name)

    #     avg_x_dict = {model_name: {} for model_name in all_x}
    #     avg_y_dict = {model_name: {} for model_name in all_y}

    #     for model_name in all_x:
    #         for test_case_name in all_x[model_name]:
    #             avg_x_dict[model_name][test_case_name] = np.mean(all_x[model_name][test_case_name])
    #             avg_y_dict[model_name][test_case_name] = np.mean(all_y[model_name][test_case_name])

    #     reformatted_x = {test_case_name: [] for test_case_name in test_file_mapping.values()}
    #     reformatted_y = {test_case_name: [] for test_case_name in test_file_mapping.values()}

    #     for model_name in avg_x_dict:
    #         for test_case_name in avg_x_dict[model_name]:
    #             reformatted_x[test_case_name].append(avg_x_dict[model_name][test_case_name])
    #             reformatted_y[test_case_name].append(avg_y_dict[model_name][test_case_name])

    #     for subskill_name, x_values in reformatted_x.items():
    #         y_values = reformatted_y[subskill_name]
    #         r_sq = calculate_OLS_linear_regression(x_values, y_values)
    #         t_test_val, p_val = t_test(x_values, y_values)
    #         print(f"Aggregated R² value for subskill name {subskill_name}: {r_sq} with p_value: {p_val}")

    #     plot_subskills(
    #         x=reformatted_x,
    #         y=reformatted_y,
    #         labels=list(all_x.keys()),
    #         title=f"Subskills Averaged Across Experiment Types",
    #         sub_titles=all_sub_titles,
    #         x_label="Average Test Case Score",
    #         y_label=metric,
    #         simulation_type=simulation_type,
    #     )

    # # all three scenarios in one plot
    # for metric in metrics:
    #     x = {}
    #     y = {}
    #     for group_name, model_dict in model_test_case.items():

    #         sub_titles = []
    #         labels = []
    #         for model_name, test_case_dict in model_dict.items():

    #             labels.append(model_name)
    #             for test_case_name, test_case_score in test_case_dict.items():
    #                 if test_case_name not in x:
    #                     x[test_case_name] = {}
    #                     y[test_case_name] = {}
    #                 if model_name not in x[test_case_name]:
    #                     x[test_case_name][model_name] = []
    #                     y[test_case_name][model_name] = []
    #                 y[test_case_name][model_name].append(model_metric_results[metric][group_name][model_name][simulation_type])
    #                 x[test_case_name][model_name].append(test_case_score)

    #                 if test_case_name not in sub_titles:
    #                     sub_titles.append(test_case_name)
    #     plot_subskills_scenarios(x=x, y=y, labels=labels, title=f"Subskills for {group_name}", sub_titles=sub_titles, x_label="Test Case Score", y_label=metric, simulation_type=simulation_type)

    # # Plot the correlations per scenario
    # for group_name, model_dict in model_test_case.items():
    #     data = {}

    #     labels = []
    #     for model_name, test_case_dict in sorted(model_dict.items(), key=lambda item: item[0]):

    #         labels.append(model_name)
    #         for test_case_name, test_case_score in sorted(test_case_dict.items(), key=lambda item: item[0]):

    #             if test_case_name not in data:
    #                 data[test_case_name] = []

    #             data[test_case_name].append(test_case_score)

    #         for metric in metrics:
    #             if metric not in data:
    #                 data[metric] = []
    #             data[metric].append(model_metric_results[metric][group_name][model_name][simulation_type])
    #     plot_correlation_per_scenario(data, group_name, simulation_type, subskill_cols=list(data.keys())[:4], metric_cols=list(data.keys())[4:])

    # plot subskills with and without universalization
    simulation_names = ["baseline", "universalization"]
    reformatted_x_baseline = None
    reformatted_y_baseline = None
    reformatted_x_universalization = None
    reformatted_y_universalization = None

    for simulation_name in simulation_names:
        all_x = {}
        all_y = {}
        all_sub_titles = []

        test_file_mapping = (
            test_file_2_test_case_baseline if simulation_name == "baseline" else test_file_2_test_case_univerlisation
        )

        for group_name, model_dict in model_test_case.items():

            for model_name, test_case_dict in model_dict.items():
                print(model_name)


                for test_file_name, test_case_name in test_file_mapping.items():
                    if test_case_name not in test_case_dict:
                        continue

                    if model_name not in all_x:
                        all_x[model_name] = {}
                        all_y[model_name] = {}

                    if test_case_name not in all_x[model_name]:
                        all_x[model_name][test_case_name] = []
                        all_y[model_name][test_case_name] = []


                    test_case_score = test_case_dict[test_case_name]
                    survival_time = model_metric_results["Mean survival time"][group_name][model_name][simulation_type]

                    all_x[model_name][test_case_name].append(test_case_score)
                    all_y[model_name][test_case_name].append(survival_time)

                    if test_case_name not in all_sub_titles:
                        all_sub_titles.append(test_case_name)

            avg_x_dict = {model_name: {} for model_name in all_x}
            avg_y_dict = {model_name: {} for model_name in all_y}

            for model_name in all_x:
                for test_case_name in all_x[model_name]:
                    avg_x_dict[model_name][test_case_name] = np.mean(all_x[model_name][test_case_name])
                    avg_y_dict[model_name][test_case_name] = np.mean(all_y[model_name][test_case_name])

            reformatted_x = {test_case_name: [] for test_case_name in test_file_mapping.values()}
            reformatted_y = {test_case_name: [] for test_case_name in test_file_mapping.values()}

            for model_name in avg_x_dict:
                for test_case_name in avg_x_dict[model_name]:
                    reformatted_x[test_case_name].append(avg_x_dict[model_name][test_case_name])
                    reformatted_y[test_case_name].append(avg_y_dict[model_name][test_case_name])

            if simulation_name == "baseline":
                reformatted_x_baseline = reformatted_x
                reformatted_y_baseline = reformatted_y
            else:
                reformatted_x_universalization = reformatted_x
                reformatted_y_universalization = reformatted_y
    print("BASELINE")
    print(reformatted_x_baseline)
    print("Baseline y")
    print(reformatted_y_baseline)
    print("UNIVERSALIATION x")
    print(reformatted_x_universalization)
    print("UNIVERSALIZATION Y")
    print(reformatted_y_universalization)
        # plot_subskills(
        #     x=reformatted_x,
        #     y=reformatted_y,
        #     labels=list(all_x.keys()),
        #     title=f"Subskills Averaged Across Experiment Types",
        #     sub_titles=all_sub_titles,
        #     x_label="Average Test Case Score",
        #     y_label=metric,
        #     simulation_type=simulation_type,
        # )

if __name__ == '__main__':
    main()
