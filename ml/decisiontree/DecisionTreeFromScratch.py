import pandas as pd
import numpy as np


def gini_index(groups, classes):
    n_instances = float(sum(len(group) for group in groups))

    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)

    return gini


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                print("get split: gini=%.3f, split_value=%.3f" % (gini, row[index]))
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    return {
        'index': b_index,
        'value': b_value,
        'groups': b_groups
    }


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def get_dataset():
    dataset_path = "./data/banknote/data_banknote_authentication.csv"
    return pd.read_csv(dataset_path, header=None, dtype=float)


if __name__ == "__main__":
    dataset = get_dataset().values

    get_split(dataset)
