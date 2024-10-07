import numpy as np
from pm4py.objects.process_tree.semantics import GenerationTree, generate_log


def extract_directly_follows_pairs(sequences):
    directly_follows_pairs = set()  # Use a set to avoid duplicate pairs

    # Iterate over each sequence
    for sequence in sequences:
        # Iterate over consecutive pairs of activities in the sequence
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i+1])  # Current activity and the next one
            directly_follows_pairs.add(pair)

    return list(directly_follows_pairs)


def compute_footprint_matrix(sequences):
    pairs = extract_directly_follows_pairs(sequences)
    return compute_footprint_matrix_pairs(pairs)


def compute_footprint_matrix_pairs(pairs, activities):    
    activities = sorted(activities)  # Sort to maintain order
    n = len(activities)
    
    # Step 2: Create an empty n x n matrix
    footprint_matrix = np.full((n, n), '#', dtype='<U2')  # Initialize with '#'
    
    # Map activities to indices
    activity_idx = {activity.strip(): idx for idx, activity in enumerate(activities)}
    
    # Step 3: Fill the matrix based on the pairs
    for a, b in pairs:
        i, j = activity_idx[a.strip()], activity_idx[b.strip()]
        footprint_matrix[i][j] = '→'  # A can follow B
    
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


def parse_tree(tree_str):
    # Parse the tree string and return a ProcessTree object TODO
    pass


def generate_traces_from_tree(tree_str, activities):
    # generate ProcessTree from string
    tree = parse_tree(tree_str)
    gen_tree = GenerationTree(tree)
    traces = generate_log(gen_tree, no_traces=100)
    string_traces = [[event["concept:name"] for event in trace] for trace in traces]
    return string_traces