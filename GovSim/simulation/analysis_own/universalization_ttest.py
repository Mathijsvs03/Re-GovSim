import os
import json
import numpy as np
from scipy.stats import ttest_rel

def extract_metrics(test):
    mean_survival = []
    gains = []
    efficiency = []
    for experiment, task in experiments:
        for model in models:
            group = f"{model}_{task}_{test}"
            if "universalization" in group:
                with open(f"results_json/{experiment}/universalization/{group}.json", 'r') as file:
                    data = json.load(file)

            else:
                with open(f"results_json/{experiment}/baseline/{group}.json", 'r') as file:
                    data = json.load(file)

            mean_survival.append(data["general"]["mean_survival"])
            gains.append(data["general"]["mean_gains"])
            efficiency.append(data["general"]["mean_efficiency"])
    return mean_survival, gains, efficiency


if __name__ == '__main__':
    experiments = [["fishing_v6.4", "fish"], ["sheep_v6.4", "sheep"], ["pollution_v6.4", "pollution"]]
    models = ["Meta-Llama-3-8B-Instruct", "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Mistral-7B-Instruct-v0.2"]
    tests = ["baseline_concurrent_universalization", "baseline_concurrent"]

    base_path = "../results/"

    survival_time_dict = {}
    gains_dict= {}
    efficiency_dict = {}
    for test in tests:
        survival_time, gains, efficiency = extract_metrics(test)
        survival_time_dict[test] = np.array(survival_time)
        gains_dict[test] = np.array(gains)
        efficiency_dict[test] = np.array(efficiency)

    difference_survival_time = survival_time_dict[tests[0]] - survival_time_dict[tests[1]]
    difference_gains = gains_dict[tests[0]] - gains_dict[tests[1]]
    difference_efficiency = efficiency_dict[tests[0]] - efficiency_dict[tests[1]]

    mean_difference_survival_time = np.mean(difference_survival_time)
    mean_difference_gains = np.mean(difference_gains)
    mean_difference_efficiency = np.mean(difference_efficiency)

    t_stat_survival_time, p_value_survival_time = ttest_rel(survival_time_dict[tests[0]], survival_time_dict[tests[1]], alternative='greater')
    t_stat_gains, p_value_gains = ttest_rel(gains_dict[tests[0]], gains_dict[tests[1]], alternative='greater')
    t_stat_efficiency, p_value_efficiency = ttest_rel(efficiency_dict[tests[0]], efficiency_dict[tests[1]], alternative='greater')

    print(f"average increase survival time {mean_difference_survival_time} with p value {p_value_survival_time} is it significant {p_value_survival_time < 0.05}")
    print(f"average increase gains {mean_difference_gains} with p value {p_value_gains} is it significant {p_value_gains < 0.05}")
    print(f"average increase efficiency {mean_difference_efficiency} with p value {p_value_efficiency} is it significant {p_value_efficiency < 0.05}")


