from math import log
from operator import itemgetter


def create_data_set():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']

    return dataset, labels


def calc_shannon_entropy(dataset):
    n = len(dataset)
    label_counts = {}

    # count the number of data sample of each label
    # so the result may be label_counts = {"yes": 2, "no": 3}
    for feature_vec in dataset:
        current_label = feature_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    '''
    Shannon Entropy of a dataset
        E(x) = sum{ -p[i] * log(p[i]) }
        p[i] is the probability of label i, for i iterates all labels
    '''
    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / n
        entropy -= prob * log(prob, 2)

    return entropy


def split_dataset(dataset, axis, value):
    # reduce the feature vec and return a sub-dataset with feature_vec[axis] == value
    data_partition = []
    for feature_vec in dataset:
        if feature_vec[axis] == value:
            reduced_feature_vec = feature_vec[:axis]
            reduced_feature_vec.extend(feature_vec[axis+1:])
            data_partition.append(reduced_feature_vec)
    return data_partition


def choose_best_feature_to_split(dataset):
    if len(dataset) == 0:
        return -1
    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannon_entropy(dataset)  # origin entropy of the dataset
    best_info_gain = 0.0
    best_feature = -1

    for axis in range(num_features):
        values = [example[axis] for example in dataset]
        unique_values = set(values)  # unique the feature values
        new_entropy = 0.0
        '''
        first calc the new entropy after splitting data by the chosen feature
        summing up entropy of all subset by weight(probability) to get the new entropy
        then, info_gain = base_entropy - new_entropy
        '''
        for value in unique_values:
            sub_dataset = split_dataset(dataset, axis, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_entropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_feature = axis
            best_info_gain = info_gain

    return best_feature


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    class_list = [sample[-1] for sample in dataset]
    # if all the sample have the same label class, then return
    if class_list.count(class_list[0]) == len(dataset):
        return class_list[0]
    # if there is no more feature to split the data, return the majority label class
    if len(dataset[0]) == 1:
        return majority_count(class_list)

    # choose a best feature to split the dataset
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]

    # tree = {<string> : <tree>}
    tree = {best_feature_label: {}}
    # delete elements from list by index
    del(labels[best_feature])
    feature_values = [sample[best_feature] for sample in dataset]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value), sub_labels)

    return tree


if __name__ == "__main__":
    data = create_data_set()

    print(data)
