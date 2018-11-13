import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


def get_some_mae():
    for max_leaf_nodes in [5, 50, 500, 5000]:
        mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("nodes: %4d, mae: %6d" % (max_leaf_nodes, mae))


def load_data():
    data_path = "./data/melb/melb_data.csv"
    data = pd.read_csv(data_path)

    filtered_data = data.dropna(axis=0)

    y = filtered_data.Price
    features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt',
                'Lattitude', 'Longtitude']

    X = filtered_data[features]
    return X, y


if __name__ == "__main__":
    X, y = load_data()

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    get_some_mae()
