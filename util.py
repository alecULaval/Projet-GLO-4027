from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

exempleFile = "exemple.txt"
resultFile = "results.txt"


def readResultFile(file=exempleFile):
    with open(file) as f:
        lines = f.readlines()
        rows = [line.split("\t")[0] for line in lines]
        return rows


def writeResultFile(data, classifier, file=resultFile):
    predictions = classifier.predict(data)
    indices = data.index

    with open(file, 'w') as f:
        for prediction, indice in zip(predictions, indices):
            f.write("{}\t{}\n".format(indice, prediction))


def addColumns(X, y, columns):
    best_score = 0
    column_list = columns

    for i in range(3):
        X_filtered = X[column_list]
        X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.1)
        clf = RandomForestClassifier(max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if score > best_score:
            best_score = score

    for i in range(10):
        best_column = None
        count = 0
        X_columns = [col for col in X.columns]
        random.shuffle(X_columns)
        for column in X_columns:
            if column in column_list:
                continue
            columns = column_list.copy()
            columns.append(column)
            X_filtered = X[columns]

            X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.9)
            clf = RandomForestClassifier(max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
            clf.fit(X_train, y_train)

            score = clf.score(X_test, y_test)
            if score > best_score:
                X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.9)
                clf = RandomForestClassifier(n_jobs=-1)
                clf.fit(X_train, y_train)
                score = min(clf.score(X_test, y_test), score)
                if score > best_score:
                    best_score = score
                    best_column = column
                    print("\t{}: New best! Column: {}   \tScore: {}".format(i, best_column, best_score))
            count += 1
            if count >= 10 and not (best_column is None):
                break

        if best_column is None:
            break

        column_list.append(best_column)
        print("{}: Testing complete! Column: {}   \tScore: {}".format(i, best_column, best_score))
    print("Final list of columns:")
    print("[", end='')
    for column in column_list:
        print('"{}", '.format(column), end='')
    print("]")
    return column_list

def removeColumns(X, y, columns= None):
    best_score = 0

    column_list = columns

    X_filtered = X[column_list]
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.1)
    clf = RandomForestClassifier(max_depth=32, min_samples_leaf=4, n_estimators=32, n_jobs=-1)
    clf.fit(X_train, y_train)

    f_imp = clf.feature_importances_
    feat = clf.feature_names_in_
    ordre = np.argsort(f_imp)

    n = len(feat)
    feat = feat[ordre]
    column_list = feat[int(0.1*n):]

    for i in feat[:int(0.9*n)]:
        print("Removing column {}".format(i))

    return column_list.tolist()
