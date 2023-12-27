#!/usr/bin/env python3
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
    # try:
    #     return float(v)
    # except ValueError:
    #     try:
    #         return int(v)
    #     except ValueError:
    #         return v


def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result)
    """
    return dict(collections.Counter(row[-1] for row in part))

    # results = collections.Counter()
    # for row in part:
    #     c = row[-1]
    #     results[c] += 1
    # return dict(results)

    # results = {}
    # for row in part:
    #     c = row[-1]
    #     if c not in results:
    #         results[c] = 0
    #     results[c] += 1
    # return results


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
    raise NotImplementedError


def divideset(part: Data, column: int, value) -> Tuple[Data, Data]:
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    split_function = None
    if isinstance(value, int) or isinstance(value, float):  # for numerical values
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    set1 = [row for row in part if split_function(row)]
    set2 = [row for row in part if not split_function(row)]
    return set1, set2


class DecisionNode:
        def __init__(self, col: object = -1, value: object = None, results: object = None, tb: object = None, fb: object = None) -> object:
            """
            Initialize a Decision Node.

            :param col: the column index of the criterion to be tested
            :param value: the value that the column must match to get a true result
            :param results: stores a dictionary of results for a leaf node (None for decision nodes)
            :param tb: true branch (DecisionNode that represents the data where the condition is true)
            :param fb: false branch (DecisionNode that represents the data where the condition is false)
            """
            self.col = col
            self.value = value
            self.results = results  # None for decision nodes
            self.tb = tb  # true branch
            self.fb = fb  # false branch


def buildtree(self, part: Data, scoref=entropy, beta=0):
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0: return DecisionNode()  # cas base
    current_score = scoref(part)

    # variables

    best_gain = 0
    best_criteria = None
    best_sets = None

    column_count = len(part[0]) - 1  # count of attributes/columns
    for col in range(0, column_count):
        # Generate the list of different values in this column
        column_values = {}
        for row in part:
            column_values[row[col]] = 1

        # Now try dividing the rows up for each value in this column
        for value in column_values.keys():
            (set1, set2) = divideset(part, col, value)  # Defined function to divide the dataset

            # Information gain
            p = float(len(set1)) / len(part)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # Create the subbranches
    if best_gain > beta:
        trueBranch = buildtree(best_sets[0], scoref, beta)
        falseBranch = buildtree(best_sets[1], scoref, beta)
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        return DecisionNode()


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


def classify(tree, values):
    """Once we have the tree we will be able to classify new data using it.
   Implement the function classify(tree, row)
    that returns the label predicted for row.
     The criterion to assign the label in leaves that have multiple labels must
     be justified in the report."""


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
    tree = buildtree(data)
    print_tree(tree, headers)


if __name__ == "__main__":
    main()
