<!-- # GovSim: Governance of the Commons Simulation


![GovSim overview](imgs/govsim_pull_figure.png)

<p align="left">Fig 1: Illustration of the GOVSIM benchmark. AI agents engage in three resource-sharing scenarios: fishery, pasture, and pollution. The outcomes are cooperation (2 out of 45 instances) or collapse (43 out of 45 instances), based on 3 scenarios and 15 LLMs.
</p>

This repository accompanies our research paper titled "**Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents**"

#### Our paper:

"**[Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents](https://arxiv.org/abs/2404.16698)**" by *Giorgio Piatti\*, Zhijing Jin\*, Max Kleiman-Weiner\*, Bernhard Schölkopf, Mrinmaya Sachan, Rada Mihalcea*.

**Citation:**


```bibTeX
@misc{piatti2024cooperate,
      title={Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents},
      author={Giorgio Piatti and Zhijing Jin and Max Kleiman-Weiner and Bernhard Schölkopf and Mrinmaya Sachan and Rada Mihalcea},
      year={2024},
      eprint={2404.16698},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
``` -->


## Reproducibility study on GovSim: Governance of the Commons Simulation

The basis code of this repositry is provided by *Giorgio Piatti*, *Zhijing Jin*, *Max Kleiman-Weiner*, *Bernhard Schölkopf*, *Mrinmaya Sachan*, *Rada Mihalcea*. This reproducibility study is based on their paper: "**Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents**"



## Simulation

Each experiment is defined by hydra configuration. To run an experiment, use
`python3 -m simulation.main experiment=<scenario_name>_<experiment_name>`.
For example, to run the experiment `fish_baseline_concurrent` , use
`python3 -m simulation.main experiment=fish_baseline_concurrent`. See below for the list of experiments and their ids.

```
python3 -m simulation.main experiment=<experiment_id> llm.path=<path_to_llm>
```



### Table of experiments
| Experiment in the original paper      | Fishery  | Pasture | Pollution |
| ------------------------------------ |---------------- |-------------------- | -------------- |
| Default setting   |     fish_baseline_concurrent         |      sheep_baseline_concurrent       | pollution_baseline_concurrent |
| Introuducing universalization | fish_baseline_concurrent_universalization | sheep_baseline_concurrent_universalization | pollution_baseline_concurrent_universalization |
| Ablation: no language | fish_perturbation_no_language | sheep_perturbation_no_language | pollution_perturbation_no_language |
| Greedy newcomer | fish_perturbation_outsider | - | - |


In addition to the experiments presented in the original paper, we have incorporated several of our own:


| Experiments we added      | Fishery  | Pasture | Pollution |
| ------------------------------------ |---------------- |-------------------- | -------------- |
| Consequentialism   |     fish_baseline_concurrent_consequentialism         |      sheep_baseline_concurrent_consequentialism       | fish_baseline_concurrent_consequentialism |
| Deontology | fish_baseline_concurrent_deontology | sheep_baseline_concurrent_deontology | pollution_baseline_concurrent_deontology |
| Virtue ethics | fish_baseline_concurrent_virtue_ethics | sheep_baseline_concurrent_virtue_ethics | pollution_baseline_concurrent_virtue_ethics |
| Utilitarianism | fish_baseline_concurrent_utilitarianism | sheep_baseline_concurrent_utilitarianism | pollution_baseline_concurrent_utilitarianism |
| Maximin-principle | fish_baseline_concurrent_maximin_principle | sheep_baseline_concurrent_maximin_principle | pollution_baseline_concurrent_maximin_principle |
| Universalization (without sustainability calculation) | fish_baseline_concurrent_universalization_advice | sheep_baseline_concurrent_universalization_advice | pollution_baseline_concurrent_universalization_advice |
| Expert advice | fish_baseline_concurrent_instruction | sheep_baseline_concurrent_instruction | pollution_baseline_concurrent_instruction |

## Subskills

To run the subskill evaluation, use the following command:

```
python3 -m subskills.<scenario_name>.run llm.path=<path_to_llm>
```

## Supported LLMs
The original authors state that in principle any LLM model can be used. We tested the following models:

- Mistral: `mistralai/Mistral-7B-Instruct-v0.2`
- Llama-2: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`
- Llama-3: `meta-llama/Meta-Llama-3-8B-Instruct`
- DeepSeek R1: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`


For inference we use the `pathfinder` library which is provided by the original authors. The `pathfinder` library is a prompting library, that
wraps around the most common LLM inference backends (OpenAI, Azure OpenAI, Anthropic, Mistral, OpenRouter, `transformers` library and `vllm`) and allows for inference with LLMs, it is available [here](https://github.com/giorgiopiatti/pathfinder). Note that running inference with models such as Mistral, Llama-2, and Llama-3 using the transformers library requires access to their gated repositories. This requires logging in with a Huggingface token with access permission (see [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login) for a tutorial).

We only support the `transformers` library and do not provide compatibility with `vllm`. This decision is based on our use of the `transformers` backend, which required modifications to function correctly, as it did not work seamlessly in its original form as provided by the authors.


## Data Generation

### Simulation
To execute all simulations using GovSim, we have developed a script that automates the generation of results for all models, with the exception of the DeepSeek R1 LLMs. This script, run_simulation_experiment.py, enables users to specify the number of random seeds, the model to be evaluated, and the experiment type to be conducted. By default, we provide the configuration used in our experiments. It is important to note that executing this script is computationally intensive, as it systematically evaluates all combinations of LLMs and experiment types.

To run the script, use the following command within the GovComGPTQ Conda environment:

    python3 run_simulation_experiment.py

For the DeepSeek R1 LLMs, we have developed a separate script due to the requirement for manual adjustment of the max_tokens parameter (see the header comment in run_DS_distill_experiment.py). To execute this script, use the following command within the GovComGPTQ Conda environment:

    python3 run_DS_distill_experiment.py

### Subskills
As with the simulation, we created a script to generate the results of subskill tests for a specific scenario. The arguments for this script allow users to specify the experiment type and the number of random seeds used for reproducibility.

Arguments:
- `<experiment>`: The name of the subskill experiment to run. Choices:
  - `fishing`
  - `pollution`
  - `sheep`
- `--num_seeds <int>`: (Optional) The number of random seeds to generate. Default is 3.

To execute the subskill test script, use the following command:

    python3 run_subskill_experiment <experiment> --num_seeds <number_of_seeds>

We have executed this file using the following commands:
`python3 run_subskill_experiment fishing --num_seeds 3`,
`python3 run_subskill_experiment pollution --num_seeds 3`,
`python3 run_subskill_experiment sheep --num_seeds 3`.


## Preprocssing
We have written a script to preprocess the result code from the `simulation/results` directory. Our goal is to create smaller JSON files that are easier to read. All scripts in the `simulation/analysis_own` directory should be run from this directory, meaning you first need to execute the following command:

    cd simulation/analysis_own

The script for preprocessing the results can be executed to specify specific model-test pairs. These are added as arguments in the form of the identifiers within double quotes, with multiple versions seperated with commas. The script will try to preprocess the data of all permutations of these two lists. The command looks like the following:

    python3 metrics.py <models> <tests>

We ran this script with the following configurations: `python3 metrics.py --all` and `python3 metrics.py "DeepSeek-R1-Distill-Llama-8B,DeepSeek-R1-Distill-Qwen-14B" "baseline_concurrent"`. We were not able to perform all tests with the DeepSeek R1 models, which is why we used two configurations.

After running the `metrics.py` script, we need to add the runtime to the JSON files. To do this, we retrieve the runtime through wandb. We created a script to retrieve the runtime, which can be executed with the following command:

    python3 runtimes.py

Note that you need our wandb API key to retrieve the runtime data, and we provide all the data in the `results_json` directory.


## Tables and plots generation

### Plots
We have created a script that generates the subskill plots. This script must be executed in its parent directory. It can be run with the following command:

    python3 plot_subskills.py

The script takes two arguments: `simulation_path`, which contains the simulation results, and `simulation_type`, which specifies the type of simulation. We ran the following configurations of the script: `python3 plot_subskills.py` and `python3 plot_subskills.py --simulation_type universalization`. The plots are stored in the `simulation/analysis_own/figs` directory.

We also created a script for generating the reasoning plots. This script can be executed with the following command:

    python3 reasoning_plots.py

The reasoning plots are stored in the `simulation/analysis_own/figs` directory.

### Tables
All table scripts store the tables in the `simulation/analysis_own/tables` directory.
The runtime table can be created using the following command:

    python3 run_time_table.py

We also wrote a script to generate the LaTeX tables used for each experiment type (fishing, sheep, and pollution). This script can be executed with the following command:

    python3 tables.py

We ran two configurations of this Python program: `python3 tables.py --all` and `python3 tables.py "Meta-Llama-3-8B-Instruct,Llama-2-7b-chat-hf,Llama-2-13b-chat-hf,Mistral-7B-Instruct-v0.2"`

Additionally, we created a script to generate the aggregated table over the three different scenarios. This script can be executed with the following command:

    python3 tables_aggregate.py


## Human interaction

Human interaction can be enabled by modifying the `fish_baseline_concurrent.yaml` file stored at `simulation/scenarios/fishing/conf/experiment/fish_baseline_concurrent.yaml`. The `personas:persona_0` type must be changed from `ai_agent` to `human_agent`. This changes persona 0 from an AI agent to a human interaction agent. Once the simulation is run, human interaction will be enabled. Note that we only support the fishing scenario with human interaction.

## Graphical user interface

The original authors developed a graphical user interface (GUI) to facilitate the analysis and interpretation of the results. This interface can be launched using the following command:

    python3 -m simulation.analysis.app

Navigation within the application is performed via the search bar. To access the details of a specific group, users must navigate to the simulation directory, followed by the group-specific directory, and append /details to the search string. For instance, the following URL provides access to the details of a specific simulation group:

    http://0.0.0.0:8050/fishing_v6.4/DeepSeek-R1-Distill-Llama-8B_fish_baseline_concurrent/details

This structured approach allows users to efficiently examine group-specific data within the simulation environment.

## Code Setup
A conda installation is required for the setup. If you do not have conda installed, you can install it by following the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). To use the codes in this repo, first clone this repo:


    git clone --recurse-submodules https://github.com/giorgiopiatti/GovSim.git
    cd govsim

Then, create a conda environment based on the provided `environment.yml` file:

    conda env create -f environment.yml


