"""
This script generates and saves a plot comparing the performance of different models
across various reasoning strategies in multiple scenarios. The performance metric is
specified by the user and the results are aggregated over the scenarios.

Functions:
    get_results(path, metric):
        Reads a JSON file from the given path and returns the specified metric from the "general" section.

    make_plot(models, reasonings, metric, scenario="aggregated", plot_info={'t': "Title", 'f': "FolderName", 'n': "FileName"}):
        Generates and saves a scatter plot comparing the performance of different models
        across various reasoning strategies. The plot is saved to the specified folder
        with the given filename.

Usage:
    The script can be run as a standalone program. It will generate a plot comparing
    the mean gains of different models across various reasoning strategies and save
    the plot to the specified folder.

"""

import matplotlib.pyplot as plt
import json

def get_results(path, metric):
    with open(path, 'r') as file:
        print(path)
        temp_res = json.load(file)
        return temp_res.pop("general")[metric]

def make_heatmap(models, reasonings):
    metrics = ["mean_gains", "mean_survival", "mean_efficiency", "mean_equality", "mean_over_usage"]
    # social reasoning on the x-as and scenario on the y-as en dat per metric en per model

    scen_dict = {"fishing_v6.4": "fish",
                 "sheep_v6.4": "sheep",
                 "pollution_v6.4": "pollution"}

    scenarios = ["fishing_v6.4", "sheep_v6.4", "pollution_v6.4"]
    for m in models:
        for metric in metrics:
            # op de y-as
            heatmap = []
            for s in scenarios:
                # op de x-as
                scenario_results = []
                for r in reasonings:
                    exp = "_" + r if not r == "baseline" else ""
                    path = f"results_json/{s}/{r}/" + \
                        f"{m}_{scen_dict[s]}_baseline_concurrent{exp}.json"
                    scenario_results.append(get_results(path, metric))
                heatmap.append(scenario_results)
            plt.title(f"model: {m} with metric: {metric}")
            plt.imshow(np.array(heatmap))
            plt.show()



def make_plot(models, reasonings, metric, scenario="aggregated",
              plot_info={'t': "Title", 'f': "FolderName", 'n': "FileName"}):
    # Obtain metric scores
    scenarios = ["fishing_v6.4", "sheep_v6.4", "pollution_v6.4"] if \
        scenario == "aggregated" else [scenario]
    scen_dict = {"fishing_v6.4": "fish",
                 "sheep_v6.4": "sheep",
                 "pollution_v6.4": "pollution"}

    total_res = []
    for m in models:
        model_res = []
        for r in reasonings:
            reason_mean = 0
            for s in scenarios:
                exp = "_" + r if not r == "baseline" else ""
                path = f"results_json/{s}/{r}/" + \
                    f"{m}_{scen_dict[s]}_baseline_concurrent{exp}.json"
                reason_mean += get_results(path, metric)
            model_res += [reason_mean / len(scenarios)]
        total_res += [model_res]

    # Plot metric scores
    reasonings_label = {"baseline": "Baseline",
                        "universalization": "Univ. with calc",
                        "universalization_advice": "Univ. no calc",
                        "deontology": "Deontology",
                        "virtue_ethics": "Virtue ethics",
                        "utilitarianism": "Utilitarianism",
                        "consequentialism": "Consequentialism",
                        "maximin_principle": "Maximin principle",
                        "instruction": "Expert advice"}
    metric = "mean gain" if metric == "mean_gains" else metric
    markers = ["X", "s", "D", "^", "*", "P", "o", "v", "<", ">"]

    x = [reasonings_label[r] for r in reasonings]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, (1, 2))

    i = 0
    for m in models:
        ax.scatter(x, total_res.pop(0), label=m, marker=markers[i % len(markers)])
        i += 1
    ax.set_title(plot_info['t'], pad=32, fontweight="bold")
    ax.set_xlabel("Injected prompts per model", fontweight="bold")
    ax.set_ylabel(f"Aggregated {metric}", fontweight="bold")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=len(models))
    plt.grid(True, axis="y", which="both")
    plt.xticks(rotation=20, ha="right")
    fig.subplots_adjust(bottom=0.25)
    fig.subplots_adjust(top=0.8)

    plt.savefig(f"{plot_info['f']}/{plot_info['n']}")

if __name__ == "__main__":
    models = ["Llama-2-7b-chat-hf",
              "Llama-2-13b-chat-hf",
              "Meta-Llama-3-8B-Instruct",
              "Mistral-7B-Instruct-v0.2"]
    reasonings = ["baseline",
                  "universalization",
                  "universalization_advice",
                  "deontology",
                  "virtue_ethics",
                  "utilitarianism",
                  "consequentialism",
                  "maximin_principle",
                  "instruction"]
    metric = "mean_gains"

    plot_dict = {'t': "Aggregated mean gain over all scenarios per model",
                 'f': "figs",
                 'n': "aggregated_gain_plot"}

    # make_plot(models, reasonings, metric, plot_info=plot_dict)

    make_heatmap(models, reasonings)