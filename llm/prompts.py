general_task_prompt = """Given a set of activities constituting an organizational process and a trace, determine whether the trace is a valid execution of the process. Each trace is a sequence of activities that must be performed in the correct order in order to be valid.
Provide either True or False as the answer and nothing else.
"""

next_activity_prompt_v1 = """
Consider an organizational process consisting of a set of activities. 
A trace represents a sequence of activities performed in a specific order within this process. 
Given a prefix of a trace, your task is to predict the next activity to be performed. 
The output should be either one of the activities from the given set or [END] if no activity should follow.
"""

next_activity_prompt = """
Consider an organizational process consisting of a set of activities. 
A trace represents a sequence of activities performed in a specific order in this process. 
Given a prefix of a trace, your task is to predict the next activity to be performed. 
The answer should be exactly one of the activities from the given set and nothing else.
"""

general_task_prompt_order = """You are given a set of activities constituting an organizational process and two activities performed in a single process execution. 
Determine whether it was valid for the first activity to occur before the second. 
Provide either True or False as the answer and nothing else."""

def get_few_shot_prompt_prefix(sample_df, n_samples, task_prompt, input_att):
    if n_samples == 0:
        return task_prompt + "Activities: "
    in_context_examples = sample_df.sample(n=n_samples*2)
    examples = "\nExamples:\n"
    for i, row in in_context_examples.iterrows():
        examples += ("All process activities: " + str(row["unique_activities"]) + "\n"
                     + "Prefix of a Trace:" + str(row['prefix']) + "\n"
                     + f"Predicted Next Activity: {row['next']}\n\n")
    few_shot_prompt = task_prompt + examples
    return few_shot_prompt + "Your task:\nAll process activities: "


def get_few_shot_prompt_pairs(sample_df, n_samples, task_prompt, input_att):
    if n_samples == 0:
        return task_prompt + "Activities: "
    in_context_positive = sample_df[~sample_df["out_of_order"]].sample(n=n_samples)
    in_context_negative = sample_df[sample_df["out_of_order"]].sample(n=n_samples)
    positive_examples = "\nExamples:\n"
    for i, row in in_context_positive.iterrows():
        positive_examples += ("All process activities: " + str(row["unique_activities"]) + "\n"
                              + "1. Activity:" + str(row[input_att][0]) + "\n"
                              + "2. Activity:" + str(row[input_att][1]) + "\n"
                              + "Valid: True\n\n")
    negative_examples = "\n"
    for i, row in in_context_negative.iterrows():
        negative_examples += ("All process activities: " + str(row["unique_activities"]) + "\n"
                              + "1. Activity:" + str(row[input_att][0]) + "\n"
                              + "2. Activity:" + str(row[input_att][1]) + "\n"
                              + "Valid: True\n\n")
    few_shot_prompt = task_prompt + positive_examples + negative_examples
    return few_shot_prompt + "Your task:\nAll process activities: "


def get_zero_shot_prompt_pairs(task_prompt, input_att):
    return task_prompt + "Activities: "


def get_few_shot_prompt_traces(sample_df, n_samples, task_prompt, input_att):
    if n_samples == 0:
        return task_prompt + "Trace: "
    in_context_positive = sample_df[~sample_df["anomalous"]].sample(n=n_samples)
    in_context_negative = sample_df[sample_df["anomalous"]].sample(n=n_samples)
    positive_examples = "\nExamples:\n"
    for i, row in in_context_positive.iterrows():
        positive_examples += "All process activities: " + str(row["unique_activities"]) + "\n" + "Trace: " + str(
            row["trace"]) + "\n" + " Valid: True\n\n"
    negative_examples = "\n"
    for i, row in in_context_negative.iterrows():
        negative_examples += "All process activities: " + str(row["unique_activities"]) + "\n" + "Trace: " + str(
            row["trace"]) + "\n" + "Valid: False\n\n"
    few_shot_prompt = task_prompt + positive_examples + negative_examples
    return few_shot_prompt + "Your task:\nAll process activities: "


def get_zero_shot_prompt_traces(task_prompt, input_att):
    return task_prompt + "Trace: "
