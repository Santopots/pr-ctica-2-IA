#!/usr/bin/env python3
import random
import sys
import collections
from math import log2
from typing import List, Tuple

# Used for typing
Data = List[List]


def read(file_name: str, separator: str = ",") -> Tuple[List[str], Data]:
    """
    t3: Load the data into a bidimensional list.
    Return the headers as a list, and the data
    """
    header = None
    data = []
    with open(file_name, "r") as fh:
        for line in fh:
            values = line.strip().split(separator)
            if header is None:
                header = values
                continue
            data.append([_parse_value(v) for v in values])
    return header, data


def _parse_value(v: str):
    try:
        if float(v) == int(v):
            return int(v)
        else:
            return float(v)
    except ValueError:
        return v


def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result)
    """
    return dict(collections.Counter(row[-1] for row in part))


def gini_impurity(part: Data):
    """
    t5: Computes the Gini index of a node
    """
    total = len(part)
    if total == 0:
        return 0

    results = unique_counts(part)
    imp = 1
    for v in results.values():
        imp -= (v / total) ** 2
    return imp


def entropy(part: Data):
    """
    t6: Entropy is the sum of p(x)log(p(x))
    across all the different possible results
    """
    total = len(part)
    results = unique_counts(part)

    probs = (v / total for v in results.values())
    return -sum(p * log2(p) for p in probs)

    # imp = 0
    # for v in results.values():
    #     p = v / total
    #     imp -= p * log2(p)
    # return imp


def _split_numeric(prototype: List, column: int, value):
    return prototype[column] >= value


def _split_categorical(prototype: List, column: int, value: str):
    return prototype[column] == value


def divideset(part: Data, column: int, value) -> Tuple[Data, Data]:
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical
        # Split "part" according "split_function"
    set1, set2 = [], []
    for row in part:  # For each row in the dataset
        if split_function(row, column, value):  # If it matches the criteria
            set1.append(row)  # Add it to the first set
        else:
            set2.append(row)  # Add it to the second set
    return (set1, set2)  # Return both sets


class DecisionNode:
    def __init__(self, col: object = -1, value: object = None, results: object = None, tb: object = None,
                 fb: object = None,
                 split_quality: object = 0) -> object:
        """
        t8: We have 5 member variables:
        - col is the column index which represents the
          attribute we use to split the node
        - value corresponds to the answer that satisfies
          the question
        - tb and fb are internal nodes representing the
          positive and negative answers, respectively
        - results is a dictionary that stores the result
          for this branch. Is None except for the leaves
          @rtype: object
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        self.split_quality = split_quality


def buildtree(self, part: Data, scoref=entropy, beta=0):
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0: return DecisionNode()  # cas base
    current_score = scoref(part)

    if current_score == 0:
        return DecisionNode(unique_counts(part), 0) #pure node

    best_gain = 0
    best_criteria = None
    best_sets = None
    column_count = len(part[0]) - 1  # count of attributes/columns

    for col in range(0, column_count):
        column_values = {}
        for row in part:
            column_values[row[col]] = 1

        for value in column_values.keys():
            (set1, set2) = divideset(part, col, value)

            p = float(len(set1)) / len(part)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > beta:
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=buildtree(best_sets[0]), fb=buildtree(best_sets[1]), split_quality=best_gain)
    else:
        return DecisionNode(results=unique_counts(part), split_quality=best_gain)

def buildtree2(part: Data, scoref=entropy, beta=0): #Recursive version
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)
    if current_score == 0:
        return DecisionNode(results=unique_counts(part), split_quality=0) # Pure node

    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(part[0]) - 1 # -1 because the last column is the label
    for col in range(0, column_count): # Search the best parameters to use
        column_values = {}
        for row in part:
            column_values[row[col]] = 1
        for value in column_values.keys():
            (set1, set2) = divideset(part, col, value)
            p = float(len(set1)) / len(part)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > beta:
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=buildtree(best_sets[0]), fb=buildtree(best_sets[1]), split_quality=best_gain)
    else:
        return DecisionNode(results=unique_counts(part), split_quality=best_gain)

def iterative_buildtree(part, scoref=entropy, beta=0):
    # Initialize the stack with the initial dataset and a dummy parent node
    stack = [(part, None, None)]

    # The root node will be set once the first split is made
    root = None

    while stack:
        # Pop a state off the stack
        part, parent, is_true_branch = stack.pop()

        if len(part) == 0: continue  # Skip empty subsets

        current_score = scoref(part)

        best_gain = 0
        best_criteria = None
        best_sets = None

        column_count = len(part[0]) - 1  # count of attributes/columns
        for col in range(0, column_count):
            # Generate the list of different values in this column
            column_values = set(row[col] for row in part)

            # Now try dividing the rows up for each value in this column
            for value in column_values:
                set1, set2 = divideset(part, col, value)

                # Information gain
                p = float(len(set1)) / len(part)
                gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)

        # Create subbranches or leaf nodes
        if best_gain > beta and best_sets:
            trueBranch = DecisionNode(col=best_criteria[0], value=best_criteria[1])
            falseBranch = DecisionNode(col=best_criteria[0], value=best_criteria[1])
            if parent:
                # Attach the new nodes to the parent
                if is_true_branch:
                    parent.tb = trueBranch
                else:
                    parent.fb = falseBranch
            else:
                # If this is the first split, set the root
                root = trueBranch

            # Push the subsets back onto the stack to continue processing
            stack.append((best_sets[0], trueBranch, True))
            stack.append((best_sets[1], falseBranch, False))
        else:
            leaf_node = DecisionNode(results=unique_counts(part))
            if parent:
                # Attach the leaf node to the parent
                if is_true_branch:
                    parent.tb = leaf_node
                else:
                    parent.fb = leaf_node
            else:
                # If this is the first node and no split is made, it's the root
                root = leaf_node

    return root


def classify(tree, row):
    """Once we have the tree we will be able to classify new data using it.
   Implement the function classify(tree, row)
    that returns the label predicted for row.
     The criterion to assign the label in leaves that have multiple labels must
     be justified in the report."""
    if tree.results is not None:
        maximum = max(tree.results.values())
        labels = [k for k, v in tree.results.items() if v == maximum]
        return random.choice(labels)
    if isinstance(tree.value, (int, float)):
        if _split_numeric(row, tree.col, tree.value):
            return classify(tree.tb, row)
        else:
            return classify(tree.fb, row)
    else:
        if _split_categorical(row, tree.col, tree.value):
            return classify(tree.tb, row)
        else:
            return classify(tree.fb, row)



def print_tree(tree, headers=None, indent=""):
    """
    t11: Include the following function
    """
    # Is this a leaf node?
    if tree.results is not None:
        print(tree.results)
    else:
        # Print the criteria
        criteria = tree.col
        if headers:
            criteria = headers[criteria]
        print(f"{indent}{criteria}: {tree.value}?")

        # Print the branches
        print(f"{indent}T->")
        print_tree(tree.tb, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.fb, headers, indent + "  ")


def print_data(headers, data):
    colsize = 15
    print('-' * ((colsize + 1) * len(headers) + 1))
    print("|", end="")
    for header in headers:
        print(header.center(colsize), end="|")
    print("")
    print('-' * ((colsize + 1) * len(headers) + 1))
    for row in data:
        print("|", end="")
        for value in row:
            if isinstance(value, (int, float)):
                print(str(value).rjust(colsize), end="|")
            else:
                print(value.ljust(colsize), end="|")
        print("")
    print('-' * ((colsize + 1) * len(headers) + 1))


def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "iris.csv"

    header, data = read(filename)
    print_data(header, data)
    print(unique_counts(data))

    # print(gini_impurity(data))
    # print(gini_impurity([]))
    # print(gini_impurity([data[0]]))

    # print(entropy(data))
    # print(entropy([]))
    # print(entropy([data[0]]))

    headers, data = read(filename)
    tree = buildtree2(data)
    print_tree(tree, headers)


if __name__ == "__main__":
    main()
