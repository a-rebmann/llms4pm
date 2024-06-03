# Evaluating the Ability of LLMs to Solve Semantics-Aware Process Mining Tasks

## About
This repository contains the evaluation scripts and results as described in the manuscript
<i>Evaluating the Ability of LLMs to Solve Semantics-Aware Process Mining Tasks</i> submitted to ICPM 2024.

## Data
The corpus and datasets can be downloaded from here: [datasets](https://zenodo.org/records/11276246)

## Results
The results of the experiments can be found in the 'eval/' folder. They are stored in a .csv file 

## Running the Experiments
### Hardware Requirements
Nvidia GPU with at least 42GB of memory is required to run the experiments (e.g., RTX A6000).

### Setup 
#### via pip

First, set up a virtual env (conda is useful for this):

```shell
pip install -r requirements.txt
```

### In-Context Learning

1. Create a folder called 'data/' one level above the project root folder.
2. Place the downloaded dataset into the 'data/' folder.
3. Place the 'train_val_test.pkl' file in the 'data/' folder.

4. To run the ICL experiments, execute the following CLI in the project root folder (set the parameters of your choice, the config of the paper corresponds to the examples listed in the parameter descriptions below):
```shell
python evaluate_llm.py --task --device --hf_model --rand_shots --runs --num_samples
```

#### Parameters
--task: one of "out_of_order", "trace_anomaly", "next_activity"

--device: e.g., "cuda:0" for the first GPU on your machine

--hf_model: the Huggingface name of the model, e.g., "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai-Mistral-7B-Instruct-v0.2", 

--rand_shots: a list with the numbers of shots to include, e.g., "[3,5]" for 6 and 10 shots (twice the amount is taken.
In case of binary tasks one positive and one negative example from the same process)

--runs: the number of runs to execute

--num_samples: the number of samples to draw from the test set, e.g., 20000

### Fine-tuning
The fine-tuning experiments are run using the Trident framework. See this repository for our fine-funing evaluation scripts and how to use them:
https://github.com/fdschmidt93/trident-bpm
- The sub-folder 'bpm' contains the necessary preprocessing and evaluation code.
- The individual tasks can be run using bash scripts:
    - 'pair.sh' for A-SAD using LLMs
    - 'trace.sh' for T-SAD using LLMs
    - 'activity.sh' for S-NAP using LLMs
    - 'trace_activity.sh' for multi-task T-SAD and S-NAP using LLMs
    - 'pair_roberta.sh' for A-SAD using RoBERTa
    - 'trace_roberta.sh' for T-SAD using RoBERTa
    - 'activity_roberta.sh' for S-NAP using RoBERTa
    - 'trace_activity_roberta.sh' for multi-task T-SAD and S-NAP using RoBERTa
