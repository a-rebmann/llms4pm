import string
import numpy as np
from pm4py.objects.process_tree.semantics import GenerationTree, ProcessTree, generate_log
import re
from pm4py.objects.process_tree.obj import Operator
from uuid import uuid4


def extract_directly_follows_pairs(sequences):
    directly_follows_pairs = set()  # Use a set to avoid duplicate pairs

    # Iterate over each sequence
    for sequence in sequences:
        # Iterate over consecutive pairs of activities in the sequence
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i+1])  # Current activity and the next one
            directly_follows_pairs.add(pair)

    return list(directly_follows_pairs)


def compute_footprint_matrix(sequences, activities):
    pairs = extract_directly_follows_pairs(sequences)
    return compute_footprint_matrix_pairs(pairs, activities)


def compute_footprint_matrix_pairs(pairs, activities):    
    activities = sorted(activities)  # Sort to maintain order
    n = len(activities)
    
    # Step 2: Create an empty n x n matrix
    footprint_matrix = np.full((n, n), '#', dtype='<U2')  # Initialize with '#'
    
    # Map activities to indices
    activity_idx = {activity.strip(): idx for idx, activity in enumerate(activities)}
    
    # Step 3: Fill the matrix based on the pairs
    for a, b in pairs:
        try:
            i, j = activity_idx[a.strip()], activity_idx[b.strip()]
            footprint_matrix[i][j] = '→'  # A can follow B
        except KeyError:
            print(f"Activity not found in the list: skipping {a} or {b}")
    
    # Step 4: Identify concurrent and opposite flows
    for i in range(n):
        for j in range(n):
            if footprint_matrix[i][j] == '→' and footprint_matrix[j][i] == '→':
                footprint_matrix[i][j] = footprint_matrix[j][i] = '‖'
            elif footprint_matrix[i][j] == '→' and footprint_matrix[j][i] == '#':
                footprint_matrix[j][i] = '←'
    return footprint_matrix


def compute_footprint_fitness(matrix1, matrix2):
    # Ensure both matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same dimensions.")
    
    total_elements = matrix1.size  # Total number of elements in the matrix
    matching_elements = np.sum(matrix1 == matrix2)  # Count matching symbols
    
    # Compute fitness as the fraction of matching symbols
    fitness = matching_elements / total_elements
    
    return fitness


# Define a class to represent each node in the tree
class TreeNode:
    def __init__(self, name, node_type):
        self.id = uuid4()
        self.name = name
        self.node_type = node_type
        self.children: list[TreeNode] = []

    def __repr__(self):
        return f"{self.name} ({self.node_type}) - {self.children}"

def parse_tree_str(tree_str):
    root_name = tree_str.split("(")[0].strip()
    # Extract the children values
    # Create the root node
    root = TreeNode(name=root_name, node_type=root_name if root_name in ["+", "->", "x", "*"] else "activity") 
    if root.name in ["+", "->", "x", "*"]:
        children_list = []
        # remove the outermost brackets and then check for subtrees, parts that are separated by commas and have balanced brackets
        tree_str = tree_str[len(root_name):-1]
        if tree_str[0] == "(":
            tree_str = tree_str[1:]
        if tree_str[-1] == ")":
            tree_str = tree_str[:]
        start = 0
        depth = 0
        for i, c in enumerate(tree_str):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "," and depth == 0:
                children_list.append(tree_str[start:i])
                start = i + 1
        children_list.append(tree_str[start:])
        for child in children_list:
            root.children.append(parse_tree_str(child))
    return root


def convert_to_pm4py(current: TreeNode, parent: ProcessTree) -> ProcessTree:
    # node
    if current.name == "+":
        operator = Operator.PARALLEL
        label = None
    elif current.name == "->":
        operator = Operator.SEQUENCE
        label = None
    elif current.name == "x":
        operator = Operator.XOR
        label = None
    elif current.name == "*":
        operator = Operator.LOOP
        label = None
    else:
        operator = None
        label = current.name
    tree = ProcessTree(operator=operator, label=label)
    tree.parent = parent
    for child in current.children:        
        tree.children.append(convert_to_pm4py(child, parent=tree))
    return tree

def rename_nodes(tree, letter_to_activity, activities):# rename the nodes to the original activity names
    for node in tree.children:
        print(node.name)
        if node.name in letter_to_activity and node.name not in activities:
            node.name = letter_to_activity[node.name]
        rename_nodes(node, letter_to_activity, activities)
    return tree
    


def parse_tree(tree_str: str, activities: set[str]) -> ProcessTree:
    # relace all the activities with single letters
    activity_to_letter, letter_to_activity = map_activities_to_letters(activities)
    for activity, letter in activity_to_letter.items():
        if letter not in activities:
            tree_str = tree_str.replace(activity, letter)

    # remove whitespace
    tree_str = re.sub(r"\s+", "", tree_str)
    # remove quotes
    tree_str = tree_str.replace('"', "")
    tree_str = tree_str.replace("'", "")

    parsed_tree = parse_tree_str(tree_str)
    print(parsed_tree)
    # rename the nodes to the original activity names


    parsed_tree = rename_nodes(parsed_tree, letter_to_activity, activities)

    # convert the parsed tree and return a ProcessTree object 
    return convert_to_pm4py(parsed_tree, ProcessTree(operator=None, label="tree"))



def generate_traces_from_tree(tree_str, activities):
    # generate ProcessTree from string
    tree = parse_tree(tree_str, activities)
    gen_tree = GenerationTree(tree)
    traces = generate_log(gen_tree, no_traces=100)
    string_traces = [[event["concept:name"] for event in trace] for trace in traces]
    return string_traces


def map_activities_to_letters(activities):
    # List of single letters A-Z
    letters = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    # remove the 'x' letter as it is used for the xor operator
    letters.remove('x')
    letters.remove('a')
    letters.remove('X')

    # Extend with combinations if there are more than 50 activities
    if len(activities) > len(letters):
        combinations = [
            "a".join(pair)
            for pair in zip(
                letters * len(letters),
                letters * (len(activities) // len(letters) + 1),
            )
        ]
        letters.extend(combinations)

    # Create a mapping dictionary
    activity_to_letter = {
        activity: letters[i] for i, activity in enumerate(activities)
    }

    # and the reverse mapping
    letter_to_activity = {v: k for k, v in activity_to_letter.items()}

    return activity_to_letter, letter_to_activity