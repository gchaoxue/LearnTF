# Standardize data (0 mean, 1 stdev)
import pickle

import pandas
import numpy

from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier


# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pandas.read_csv(url, names=names)
# array = dataframe.values
#
# # separate array into input and output components
# X = array[:, 0:8]
# Y = array[:, 8]
#
# scaler = StandardScaler().fit(X)
# rescaledX = scaler.transform(X)
#
# # summarize transformed data
# numpy.set_printoptions(precision=3)
# print(pandas.DataFrame(rescaledX).describe())
# print(rescaledX[0:5, :])
#
# # kfold
# kfold = KFold(n_splits=10, random_state=7)
# model = LogisticRegression()
# scoring = 'neg_log_loss'
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print("LogLoss: %.3f (%.3f)" % (results.mean(), results.std()))
#
# # spot check
#
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
# names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# dataframe = read_csv(url, delim_whitespace=True, names=names)
# array = dataframe.values
# X = array[:, 0:13]
# Y = array[:, 13]
# kfold = KFold(n_splits=10, random_state=7)
# model = KNeighborsRegressor()
#
# scoring = 'neg_mean_squared_error'
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print("MeanSquaredError: %.3f" % results.mean())

#
# url = './ml/decisiontree/data/iris/iris.data'
# names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
#
# data_frame = read_csv(url, names=names)
# array = data_frame.values
# X = array[:, 0:4]
# Y = array[:, 4]
#
# num_trees = 10
# max_features = 4
# kfold = KFold(n_splits=10, random_state=7)
# model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
# results = cross_val_score(model, X, Y, cv=kfold)
# print(results.mean)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)