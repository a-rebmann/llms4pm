import numpy as np

general_task_prompt = """
Given a set of activities that constitute an organizational process and a sequence of activities, determine whether the sequence is a valid execution of the process. 
The activities in the sequence must be performed in the correct order for the execution to be valid.
Provide either True or False as the answer and nothing else.
"""

next_activity_prompt = """
You are given a list of activities that constitute an organizational process and a sequence of activities that have been performed in the given order.
Which activity from the list should be performed next in the sequence? 
The answer should be one activity from the list and nothing else.
"""

general_task_prompt_order = """
You are given a set of activities that constitute an organizational process and two activities performed in a single process execution. Determine whether it is valid for the first activity to occur before the second. 
Provide either True or False as the answer and nothing else.
"""


dfg_task_prompt = """
Given a list of activities that constitute an organizational process, determine all pairs of activities that can reasonably follow each other directly in an execution of this process.
Provide only a list of pairs and use only activities from the given list followed by [END].
"""

pt_task_prompt = """
Given a list of activities that constitute an organizational process, determine the process tree of the process.
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
"""

traces_task_prompt = """
Given a list of activities that constitute an organizational process, provide all possible sequences of activities, where each sequence represents a valid execution of the process.
The activities in the sequence must be performed in the correct order for the execution to be valid.
Provide only a list of activitiy sequences as the answer and nothing else.
"""

loops = """* ( A, B ) is a loop. So the process tree A is executed, then either you exit the loop, or you execute B and then A again (this can happen several times until the loop is exited)."""


def get_few_shot_prompt_dfg(sample_df, n_samples, task_prompt, input_att):
    if n_samples == 0:
        return task_prompt + "List of activities: " + str(row['unique_activities']) + "\n"
    in_context_examples = sample_df.sample(n=n_samples)
    examples = "\nExamples:\n"
    for i, row in in_context_examples.iterrows():
        # create a pair list
        pair_list = ""
        for pair in row['dfg']:
            pair_list += f"{pair[0]} -> {pair[1]}\n"
        examples += ("List of activities:\n" + str(row['unique_activities']) + "\n"
                     + "Pairs of activities:\n" + pair_list + "\n[END]\n")
    few_shot_prompt = task_prompt + examples
    return few_shot_prompt + "List of activities:\n"


def get_few_shot_prompt_pt(sample_df, n_samples, task_prompt, input_att):
    if n_samples == 0:
        return task_prompt + "List of activities: " + str(row['unique_activities']) + "\n"
    in_context_examples = sample_df.sample(n=n_samples)
    examples = "\nExamples:\n"
    for i, row in in_context_examples.iterrows():
        examples += ("List of activities:\n" + str(row['unique_activities']) + "\n"
                     + "Process tree:\n" + row["pt"] + "\n[END]\n")
    few_shot_prompt = task_prompt + examples
    return few_shot_prompt + "List of activities:\n"


def get_few_shot_prompt_prefix(sample_df, n_samples, task_prompt, input_att):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n_samples == 0:
        return task_prompt + "Activities: "
    in_context_examples = sample_df.sample(n=n_samples*2)
    examples = "\nExamples:\n"
    for i, row in in_context_examples.iterrows():
        # create an activity list with a capital letter as counter
        act_list = "0. [END]\n"
        list_of_activities = list(row['unique_activities'])
        for idx, act in enumerate(list_of_activities):
            act_list += f"{alphabet[idx]}. {act}\n"

        examples += ("List of activities:\n" + act_list + "\n"
                     + "Sequence of activities:" + str(row['prefix']) + "\n"
                     + f"Answer: {alphabet[list_of_activities.index(row['next'])] if row['next'] != '[END]' else '0'}\n\n")
    few_shot_prompt = task_prompt + examples
    return few_shot_prompt + "List of activities:\n"


def get_few_shot_prompt_pairs(sample_df, n_samples, task_prompt, input_att):
    if n_samples == 0:
        return task_prompt + "Activities: "
    model_ids = sample_df["id"].unique()
    sample_ids = np.random.choice(model_ids, n_samples)
    positive_examples = "\nExamples:\n"
    # get one positive and one negative example for each id in sample_ids
    for model_id in sample_ids:
        in_context_positive = sample_df[(sample_df["id"] == model_id) & ~sample_df["out_of_order"]].sample(n=1)
        for i, row in in_context_positive.iterrows():
            positive_examples += ("Set of process activities: " + str(row["unique_activities"]) + "\n"
                                  + "1. Activity: " + str(row[input_att][0]) + "\n"
                                  + "2. Activity: " + str(row[input_att][1]) + "\n"
                                  + "Valid: True\n")
    negative_examples = ""
    for model_id in sample_ids:
        in_context_negative = sample_df[(sample_df["id"] == model_id) & sample_df["out_of_order"]].sample(n=1)
        for i, row in in_context_negative.iterrows():
            negative_examples += ("Set of process activities: " + str(row["unique_activities"]) + "\n"
                                  + "1. Activity: " + str(row[input_att][0]) + "\n"
                                  + "2. Activity: " + str(row[input_att][1]) + "\n"
                                  + "Valid: False\n")
    few_shot_prompt = task_prompt + positive_examples + negative_examples
    return few_shot_prompt + "Set of process activities: "



    #in_context_positive = sample_df[~sample_df["out_of_order"]].sample(n=n_samples)
    #in_context_negative = sample_df[sample_df["out_of_order"]].sample(n=n_samples)
    # positive_examples = "\nExamples:\n"
    # for i, row in in_context_positive.iterrows():
    #     positive_examples += ("Set of process activities: " + str(row["unique_activities"]) + "\n"
    #                           + "1. Activity:" + str(row[input_att][0]) + "\n"
    #                           + "2. Activity:" + str(row[input_att][1]) + "\n"
    #                           + "Valid: True\n\n")
    # negative_examples = "\n"
    # for i, row in in_context_negative.iterrows():
    #     negative_examples += ("Set of process activities: " + str(row["unique_activities"]) + "\n"
    #                           + "1. Activity:" + str(row[input_att][0]) + "\n"
    #                           + "2. Activity:" + str(row[input_att][1]) + "\n"
    #                           + "Valid: True\n\n")
    # few_shot_prompt = task_prompt + positive_examples + negative_examples
    # return few_shot_prompt + "Set of process activities: "


def get_zero_shot_prompt_pairs(task_prompt, input_att):
    return task_prompt + "Set of process activities: "


def get_few_shot_prompt_traces(sample_df, n_samples, task_prompt, input_att):
    if n_samples == 0:
        return task_prompt + "Trace: "
    model_ids = sample_df["id"].unique()
    sample_ids = np.random.choice(model_ids, n_samples)
    positive_examples = "\nExamples:\n"
    # get one positive and one negative example for each id in sample_ids
    for model_id in sample_ids:
        in_context_positive = sample_df[(sample_df["id"] == model_id) & ~sample_df["anomalous"]].sample(n=1)
        for i, row in in_context_positive.iterrows():
            positive_examples += "Set of process activities: " + str(row["unique_activities"]) + "\n" + "Trace: " + str(
                row["trace"]) + "\n" + "Valid: True\n"
    negative_examples = ""
    for model_id in sample_ids:
        in_context_negative = sample_df[(sample_df["id"] == model_id) & sample_df["anomalous"]].sample(n=1)
        for i, row in in_context_negative.iterrows():
            negative_examples += "Set of process activities: " + str(row["unique_activities"]) + "\n" + "Trace: " + str(
                row["trace"]) + "\n" + "Valid: False\n"
    few_shot_prompt = task_prompt + positive_examples + negative_examples
    return few_shot_prompt + "Set of process activities: "

    # in_context_positive = sample_df[~sample_df["anomalous"]].sample(n=n_samples)
    # in_context_negative = sample_df[sample_df["anomalous"]].sample(n=n_samples)
    # positive_examples = "\nExamples:\n"
    # for i, row in in_context_positive.iterrows():
    #     positive_examples += "Set of process activities: " + str(row["unique_activities"]) + "\n" + "Trace: " + str(
    #         row["trace"]) + "\n" + " Valid: True\n\n"
    # negative_examples = "\n"
    # for i, row in in_context_negative.iterrows():
    #     negative_examples += "Set of process activities: " + str(row["unique_activities"]) + "\n" + "Trace: " + str(
    #         row["trace"]) + "\n" + "Valid: False\n\n"
    # few_shot_prompt = task_prompt + positive_examples + negative_examples
    # return few_shot_prompt + "Set of process activities: "


def get_zero_shot_prompt_traces(task_prompt, input_att):
    return task_prompt + "Set of process activities: "
