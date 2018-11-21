""" https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
"""
from _csv import reader

from random import seed
from random import randrange


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


def split(node, max_depth, min_size, depth):
    """
        node:
            index: int
            value: number
            groups:
                left: dataset
                right: dataset
    """
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    print("root: index: " + str(root['index']))
    print("root: value: " + str(root['value']))
    print("root: left_size: " + str(len(root['groups'][0])))
    print("root: right_size: " + str(len(root['groups'][1])))

    split(root, max_depth, min_size, 1)
    return root


def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset


def str_column_to_flost(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    print(dataset)
    dataset_copy = list(dataset)

    print(type(dataset_copy))
    print(dataset_copy)
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size and len(dataset_copy) > 0:
            # print("len: " + str(len(dataset_copy)))
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    return dataset_split


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)

        print(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions


if __name__ == "__main__":
    seed(1)
    filename = "./data/banknote/data_banknote_authentication.csv"
    dataset = load_csv(filename)
    for i in range(len(dataset[0])):
        str_column_to_flost(dataset, i)

    n_folds = 5
    max_depth = 5
    min_size = 10

    scores = evaluate(dataset, decision_tree, n_folds, max_depth, min_size)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

