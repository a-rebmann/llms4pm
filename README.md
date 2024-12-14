# Evaluating the Ability of LLMs to Solve Semantics-Aware Process Mining Tasks

## About
This repository contains the evaluation scripts and results as described in the manuscript
<i>On the potential of LLMs to solve semantics-ware process mining tasks</i> submitted to Process Science.

## Data
The corpus and datasets can be downloaded from here: [datasets](https://zenodo.org/records/11276246)

### ICL task descriptions (taken from llm/prompts.py)
- T-SAD: Given a set of activities that constitute an organizational process and a sequence of activities, determine whether the sequence is a valid execution of the process. 
The activities in the sequence must be performed in the correct order for the execution to be valid.
Provide either True or False as the answer and nothing else.
- A-SAD: You are given a set of activities that constitute an organizational process and two activities performed in a single process execution. Determine whether it is valid for the first activity to occur before the second. 
Provide either True or False as the answer and nothing else.Y
- S-NAP: You are given a list of activities that constitute an organizational process and a sequence of activities that have been performed in the given order.
Which activity from the list should be performed next in the sequence? 
The answer should be one activity from the list and nothing else.
- S-DFD: Given a list of activities that constitute an organizational process, determine all pairs of activities that can reasonably follow each other directly in an execution of this process.
Provide only a list of pairs and use only activities from the given list followed by [END].
- S-PTD: Given a list of activities that constitute an organizational process, determine the process tree of the process.
A process tree is a hierarchical process model.
The following operators are defined for process trees:
-> ( A, B ) tells that process tree A should be executed before process tree B
X ( A, B ) tells that there is an exclusive choice between executing process tree A and process tree B
+ ( A, B ) tells that process tree A and process treee B are executed in true concurrency.
* ( A, B ) tells that process tree A is executed, then either you exit the loop, or you execute B and then A again (this can happen several times until the loop is exited).
the leafs of a process tree are either activities or silent steps (indicated by tau).
An example process tree follows:
+ ( 'a', -> ( 'b', 'c', 'd' ) )
It defines that you should execute b before executing c and c before d. In true concurrency to this, you can execute a. Therefore, the possible traces that this tree allows for are a->b->c->d, b->a->c->d, b->c->a->d, b->c->d->a.
Provide the process tree in the format of the example as the answer followed by [END]. 
Use only activities from the given list as leaf nodes and only the allowed operators (->, X, +, *) as inner nodes. 
Also make sure each activity is used exactly once in the tree and there and each subtree has exactly one root node, i.e., pay attention to set parentheses correctly.


## Results
The results of the experiments can be found in the 'eval/' folder. 

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
--task: one of "out_of_order", "trace_anomaly", "next_activity", "dfg_generation", "pt_generation""

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
    - 'next_activity.sh' for S-DFD using LLMs (sorry for the strage naming...)
    - 'pt.sh' for S-PTD using LLMs
    - 'trace_activity.sh' for multi-task T-SAD and S-NAP using LLMs
    - 'pair_roberta.sh' for A-SAD using RoBERTa
    - 'trace_roberta.sh' for T-SAD using RoBERTa
    - 'activity_roberta.sh' for S-NAP using RoBERTa
    - 'trace_activity_roberta.sh' for multi-task T-SAD and S-NAP using RoBERTa
