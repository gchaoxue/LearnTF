""" https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
"""
from _csv import reader

from random import seed
from random import randrange
from random import sample


def gini_index(groups, classes):
    n_instances = float(sum(len(group) for group in groups))

    """
    gini-score = sum(p[i] * (1 - p[i]))
               = sum(p[i]) - sum(p[i] * p[i])
               = 1 - sum(p[i] * p[i])
    p[i] is the proportion of group i
    
    sum the scores of all groups with proportion
    """
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            p = labels.count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)

    return gini


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        # handling continuous feature
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


def get_split(dataset, feature_subsample_num):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    num_feature = len(dataset[0]) - 1
    feature_subsample_num = min(feature_subsample_num, num_feature)
    # subsample the indices
    indices = sample([i for i in range(num_feature)], feature_subsample_num)

    for index in indices:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    # print("get split: feature_index=%d, gini=%.3f, split_value=%.3f" % (b_index, b_score, b_value))

    return {
        'index': b_index,
        'value': b_value,
        'groups': b_groups
    }


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def to_score_terminal(group):
    outcomes = [row[-1] for row in group]
    distribute = dict()



def split(node, max_depth, min_size, depth, feature_subsample_num):
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
        node['left'] = node['right'] = to_score_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_score_terminal(left), to_score_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_score_terminal(left)
    else:
        node['left'] = get_split(left, feature_subsample_num)
        split(node['left'], max_depth, min_size, depth+1, feature_subsample_num)

    if len(right) <= min_size:
        node['right'] = to_score_terminal(right)
    else:
        node['right'] = get_split(right, feature_subsample_num)
        split(node['right'], max_depth, min_size, depth+1, feature_subsample_num)


def build_tree(train, max_depth, min_size, feature_subsample_num):
    root = get_split(train, feature_subsample_num)
    split(root, max_depth, min_size, 1, feature_subsample_num)
    return root


def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset


def str_column_to_float(dataset, column):
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


def predict_score(node, row):
    if row[node['index'] < node['value']]:
        if (isinstance(node['left'], dict)):
            return predict_score(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_score(node['right'], row)
        else:
            return node['right']


def predict_random_forest(forest, row):
    candidate = list()
    for tree in forest:
        candidate.append(predict(tree, row))
    return max(candidate, key=candidate.count)


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size and len(dataset_copy) > 0:
            index = randrange(len(dataset_copy))
            # fold.append(dataset_copy[index])
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
    train_scores = list()
    test_scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        prediction = algorithm(train_set, test_set, *args)
        test_actual = [row[-1] for row in fold]
        test_accuracy = accuracy_metric(test_actual, prediction["test"])
        train_actual = [row[-1] for row in train_set]
        train_accuracy = accuracy_metric(train_actual, prediction["train"])
        print("folds: train_size=%d, test_size=%d, train_accuracy=%.3f, test_accuracy=%.3f" %
              (len(train_set), len(test_set), train_accuracy, test_accuracy))
        train_scores.append(train_accuracy)
        test_scores.append(test_accuracy)
    return {"train": train_scores, "test": test_scores}


def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size, len(train[0]) - 1)
    test_predictions = list()
    train_predictions = list()
    for row in test:
        prediction = predict(tree, row)
        test_predictions.append(prediction)
    for row in train:
        prediction = predict(tree, row)
        train_predictions.append(prediction)
    return {"train": train_predictions, "test": test_predictions}


def build_forest(train, max_depth, min_size, num_tree,
                 train_subsample_num, feature_subsample_num):
    # build a list of trees
    forest = list()
    for i in range(num_tree):
        # subsample the training set
        train_size = len(train)
        indices = sample([i for i in range(train_size)], train_subsample_num)
        sub_train = list()
        for index in indices:
            sub_train.append(train[index])
        # build tree with feature sub-sampling
        tree = build_tree(sub_train, max_depth, min_size, feature_subsample_num)
        forest.append(tree)
    return forest


def random_forest(train, test, max_depth, min_size, num_tree,
                  train_subsample_num, feature_subsample_num):
    forest = build_forest(train, max_depth, min_size, num_tree,
                          train_subsample_num, feature_subsample_num)
    test_predictions = list()
    train_predictions = list()

    for row in test:
        prediction = predict_random_forest(forest, row)
        test_predictions.append(prediction)
    for row in train:
        prediction = predict_random_forest(forest, row)
        train_predictions.append(prediction)
    return {"train": train_predictions, "test": test_predictions}


def dataset_describer(dataset):
    label_values = [row[-1] for row in dataset]
    label_classes = set(label_values)

    label_class_counts = list()
    for label_class in label_classes:
        label_class_counts.append(label_values.count(label_class))
    return {
        "labels": label_classes,
        "label_counts": label_class_counts
    }


if __name__ == "__main__":
    seed(1)
    filename = "./data/banknote/data_banknote_authentication.csv"
    dataset = load_csv(filename)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    num_feature = len(dataset[0]) - 1
    n_folds = 4
    max_depth = 5
    min_size = 50
    num_tree = 15
    train_subsample_num = int(len(dataset) / 5 * 4 * 0.8)
    feature_subsample_num = int((len(dataset[0])-1) * 0.8)

    labels = dataset_describer(dataset)
    print("data number of rows: %d" % len(dataset))
    print("data number of features: %d" % num_feature)
    print("labels: %s" % labels["labels"])
    print("label counts: %s" % labels["label_counts"])
    print("folds number: %d" % n_folds)
    print("size of each fold: %d" % (len(dataset) / n_folds))
    print("max depth: %d" % max_depth)
    print("min size: %d" % min_size)
    print("train sample num: %d" % train_subsample_num)
    print("feature sample num: %d" % feature_subsample_num)

    # decision tree evaluate
    scores = evaluate(dataset, decision_tree, n_folds, max_depth, min_size)
    print('decision tree train scores: %s' % scores["train"])
    print('decision tree test scores:  %s' % scores["test"])
    print('decision tree mean accuracy: train=%.3f%%, test=%.3f%%' %
          (sum(scores["train"]) / float(len(scores["train"])),
           sum(scores["test"]) / float(len(scores["test"]))))

    # random forest
    scores = evaluate(dataset, random_forest, n_folds, max_depth, min_size, num_tree,
                      train_subsample_num, feature_subsample_num)
    print('random forest train scores: %s' % scores["train"])
    print('random forest test scores:  %s' % scores["test"])
    print('random forest mean accuracy: train=%.3f%%, test=%.3f%%' %
          (sum(scores["train"]) / float(len(scores["train"])),
           sum(scores["test"]) / float(len(scores["test"]))))

