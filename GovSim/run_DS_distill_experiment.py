"""
Script for Running DeepSeek R1 Distill agent Simulations with Configurable Parameters

WARNING:
Before running this script, ensure that you update the max tokens value in `models.py`, the path is: `simulation/utils/models.py`:
- In def gen, set max_tokens = 20000
- In def find_rule_150, set max_tokens = 1000

This script automates the process of running experiments with different LLMs by:
1. Generating random seeds for experiment reproducibility.
2. Modifying a YAML configuration file with updated model paths, group names, and seeds.
3. Executing the simulation script with the specified configuration.

Key functionalities:
- `generate_random_seeds(num_seeds)`: Generates a list of random seeds.
- `change_config(file_path, new_group_name, new_llm_path, new_seed)`: Updates the config file with new parameters.
- `run_experiment(model, seed, experiment)`: Executes an experiment using a subprocess call.
- `main()`: Iterates over predefined models and experiments, updating configurations and running simulations.

Models included:
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-Qwen-14B

Experiments included:
- Fish, Sheep, and Pollution Baseline experiments (standard & universalization variants).

Usage:
Run this script directly to execute all configured experiments:
    python3 script.py
"""

import yaml
import random
import subprocess


def generate_random_seeds(num_seeds):
    seed = 42
    ran_obj = random.Random(seed)
    seeds = [ran_obj.randint(0, 2 ** 32 - 1) for _ in range(num_seeds)]
    return seeds


def change_config(file_path, new_group_name, new_llm_path, new_seed):
    """
    Updates the `group_name`, `llm.path`, and `seed` variables in a YAML config file.

    Parameters:
        file_path (str): Path to the `config.yaml` file.
        new_group_name (str): New value for the `group_name` field.
        new_llm_path (str): New value for the `llm.path` field.
        new_seed (int): New value for the `seed` field.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        config['group_name'] = new_group_name
        config['llm']['path'] = new_llm_path
        config['seed'] = new_seed

        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)

        print("Config updated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def run_experiment(model, seed, experiment):
    command = f"python3 -m simulation.main experiment={experiment}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the experiment: {e} with model {model} and seed {seed}")

def main(experiment_configs=None):
    # Default configuration if no experiment_configs provided
    if experiment_configs is None:
        models = {
            'DeepSeek-R1-Distill-Llama-8B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
            'DeepSeek-R1-Distill-Qwen-14B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
        }
        experiments = ['fish_baseline_concurrent', 'sheep_baseline_concurrent', 'pollution_baseline_concurrent',
                      'fish_baseline_concurrent_universalization', 'sheep_baseline_concurrent_universalization',
                      'pollution_baseline_concurrent_universalization']

        for model in models:
            for experiment in experiments:
                seeds = generate_random_seeds(5)
                for seed in seeds:
                    group_name = f'{model}_{experiment}'
                    llm_path = models[model]
                    change_config('simulation/conf/config.yaml', group_name, llm_path, seed)
                    run_experiment(model, seed, experiment)
    else:
        # Process the provided experiment configurations
        model_paths = {
            'DeepSeek-R1-Distill-Llama-8B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
            'DeepSeek-R1-Distill-Qwen-14B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
            'DeepSeek-R1-Distill-Qwen-32B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
        }

        scenario_dict = {
            'fishing_v6.4': 'fish',
            'sheep_v6.4': 'sheep',
            'pollution_v6.4': 'pollution'
        }

        for model, experiment, scenario, num_runs in experiment_configs:
            seeds = generate_random_seeds(num_runs)
            for seed in seeds:
                # Combine experiment and scenario for the full experiment name
                full_experiment = f'{scenario_dict[scenario]}_{experiment}'
                group_name = f'{model}_{full_experiment}'
                llm_path = model_paths[model]

                change_config('simulation/conf/config.yaml', group_name, llm_path, seed)
                run_experiment(model, seed, full_experiment)

if __name__ == "__main__":
    # Example usage with experiment configs
    example_configs = [
        ('DeepSeek-R1-Distill-Llama-8B', 'baseline_concurrent_consequentialism', 'fishing_v6.4', 1),
        ('DeepSeek-R1-Distill-Llama-8B', 'baseline_concurrent_universalization', 'fishing_v6.4', 1),
        ('DeepSeek-R1-Distill-Llama-8B', 'baseline_concurrent_deontology', 'fishing_v6.4', 4)
    ]
    main(example_configs)