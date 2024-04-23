general_task_prompt = """
A trace is a sequence of activities that were performed the given order in an organizational process.
Given such trace and the set of all activities that can be performed in the context of the process, decide if the trace contains anomalous process behavior.
Anomalous behavior includes missing activities, superfluous activities, and activities that are performed in the wrong order.
The answer should be either True or False and nothing else.
"""

general_task_prompt_order = """
Consider an organizational process, which comprises a set of activities.
You are given the full set of activities for context.
You are also given two activities that were performed in a single execution of this process.
Decide if the following statement is True or False: the first activity was allowed to be performed before the second one. 
The answer should be either True or False and nothing else.
"""


def get_few_shot_prompt_pairs(sample_df, n_samples, task_prompt, input_att):
    if n_samples == 0:
        return task_prompt + "Activities: "
    in_context_positive = sample_df[~sample_df["anomalous"]].sample(n=n_samples)
    in_context_negative = sample_df[sample_df["anomalous"]].sample(n=n_samples)
    positive_examples = "\nExamples:\n"
    for i, row in in_context_positive.iterrows():
        positive_examples += "All process activities: " + str(
            row["unique_activities"]) + "\n" + "Activities: " + str(row[input_att]).replace("(","First activity, second activity:").replace(")","") + "\n" + "True\n\n"
    negative_examples = "\n"
    for i, row in in_context_negative.iterrows():
        negative_examples += "All process activities: " + str(
            row["unique_activities"]) + "\n" + "Activities: " + str(row[input_att]).replace("(","First activity, second activity:").replace(")","") + "\n" + "False\n\n"
    few_shot_prompt = task_prompt + positive_examples + negative_examples
    return few_shot_prompt + "All process activities: "


def get_zero_shot_prompt_pairs(task_prompt, input_att):
    return task_prompt + "Activities: "


def get_few_shot_prompt_traces(sample_df, n_samples, task_prompt, input_att):
    if n_samples==0:
        return task_prompt + "Trace: "
    in_context_positive = sample_df[sample_df["anomalous"]].sample(n=n_samples)
    in_context_negative = sample_df[~sample_df["anomalous"]].sample(n=n_samples)
    positive_examples = "\nExamples:\n"
    for i, row in in_context_positive.iterrows():
        positive_examples += "Trace: " + str(row["trace"]) + "\n" + "Known process activities: " + str(row["unique_activities"]) + "\n" + "True\n\n"
    negative_examples =  "\n"
    for i, row in in_context_negative.iterrows():
        negative_examples += "Trace: " + str(row["trace"]) + "\n" + "Known process activities: " + str(row["unique_activities"]) + "\n" + "False\n\n"
    few_shot_prompt = task_prompt + positive_examples + negative_examples
    return few_shot_prompt + "Trace: "


def get_zero_shot_prompt_traces(task_prompt, input_att):
    return task_prompt + "Trace: "