"""
This script retrieves runtime information from Weights & Biases (wandb) runs and updates corresponding JSON files with the runtime and creation date of each run.

The script performs the following steps:
1. Connects to the wandb API with a timeout of 30 seconds.
2. Retrieves runs from the "EMS" project, excluding those tagged with "skip".
3. Iterates through each run and extracts relevant information such as run name, runtime, creation date, experiment name, scenario name, and group name.
4. Determines the reasoning type based on the configuration of the run.
5. Attempts to open the corresponding JSON file and update it with the runtime and creation date.
6. Handles errors such as missing keys or file not found, and counts the total number of errors encountered.

Usage:
- Ensure you have the necessary permissions and API keys to access the wandb project.
- Run the script in an environment where the required dependencies are installed.
"""

import wandb
import json

api = wandb.Api(timeout=30)
runs = api.runs("EMS", filters={"tags": {"$nin": ["skip"]}})

errs = 0

for run in runs:
    try:
        run_name = run.name
        run_runtime = run.summary['_wandb']['runtime']
        run_created_at = run.createdAt

        experiment_name = run.config['experiment']["name"]
        scenario_name = run.config['experiment']["scenario"] + "_" + run.config['code_version']
        group_name = run.config["group_name"]

        try:
            univ = run.config['experiment']["env"]["inject_universalization"]
            if univ:
                reasoning_type = "universalization"
            else:
                reasoning_type = "baseline"

        except KeyError:
            reasoning_type = run.config['experiment']["env"]["inject_social_reasoning"]

        try:
            with open(f"results_json/{scenario_name}/{reasoning_type}/{group_name}.json", "r") as f:
                data = json.load(f)
                data[run_name]["runtime"] = run_runtime
                data[run_name]["created_at"] = run_created_at

            with open(f"results_json/{scenario_name}/{reasoning_type}/{group_name}.json", "w") as f:
                json.dump(data, f, indent=4)

        except FileNotFoundError:
            print(f"File error with {run.name} from {scenario_name}/{reasoning_type}/{group_name}")

    except KeyError:
        print(f"Error with {run.name}")
        errs += 1

print(f"Total errors: {errs}")