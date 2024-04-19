import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score

Number_trials = 10


def plot_features(df, column, num):
    label = ' '.join(column.split('_'))
    df[column].value_counts().plot(
        kind='bar', ylabel='Count',
        xlabel=f'{label}', rot=0,
        title=f'Figure {num}. {label} Distribution')
    plt.show()

    
def train_knn(X, y):
    start_time = time.time()
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []

    for seed in range(Number_trials):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        neighbors_settings = range(1, 20)
        acc_train = []
        acc_test = []
        recall = []
        precision = []

        for n_neighbors in neighbors_settings:
            clf = KNeighborsClassifier(
                n_neighbors=n_neighbors)  # build the model
            clf.fit(X_train, y_train)
            acc_train.append(clf.score(X_train, y_train))
            acc_test.append(clf.score(X_test, y_test))
            recall.append(recall_score(y_test, clf.predict(X_test)))
            precision.append(precision_score(y_test, clf.predict(X_test)))

        score_train.append(acc_train)
        score_test.append(acc_test)
        recall_test.append(recall)
        precision_test.append(precision)

    train_score = np.mean(score_train, axis=0)
    score = np.mean(score_test, axis=0)
    score_recall = np.mean(recall_test, axis=0)
    score_precision = np.mean(precision_test, axis=0)
    run_time = (time.time() - start_time)
    return ['kNN',
            np.amax(score), score_recall[np.argmax(score)+1],
            score_precision[np.argmax(score)+1],
            'N_Neighbor = {0}'.format(np.argmax(score)+1),
            'NA',
            run_time]


def train_logistic(X, y, reg):
    start_time = time.time()
    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.75,
         1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []

    for seed in range(Number_trials):
        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for alpha_run in C:
            if reg == 'l1':
                lr = LogisticRegression(
                    C=alpha_run, penalty=reg, solver='liblinear').fit(X_train, y_train)
            if reg == 'l2':
                lr = LogisticRegression(
                    C=alpha_run, penalty=reg, dual=False).fit(X_train, y_train)

            training_accuracy.append(lr.score(X_train, y_train))
            test_accuracy.append(lr.score(X_test, y_test))
            recall.append(recall_score(y_test, lr.predict(X_test)))
            precision.append(precision_score(y_test, lr.predict(X_test)))
            coefs = lr.coef_
            # append all the computed coefficients per trial
            weighted_coefs.append(coefs)

        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
        recall_test.append(recall)
        precision_test.append(precision)

    # get the mean of the weighted coefficients over all the trials
    mean_coefs = np.mean(weighted_coefs, axis=0)
    score_recall = np.mean(recall_test, axis=0)
    score_precision = np.mean(precision_test, axis=0)
    score = np.mean(score_test, axis=0)

    # Plot the weight of the parameters
    top_predictor = X.columns[np.argmax(np.abs(mean_coefs))]
    abs_mean_coefs = np.abs(mean_coefs[0, :])
    coefs_count = len(abs_mean_coefs)
    #fig, ax = plt.subplots(figsize=(3, 7))
    #ax.barh(np.arange(coefs_count), sorted(abs_mean_coefs))

    #ax.set_yticks(np.arange(coefs_count))
    #ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])

    run_time = (time.time() - start_time)
    return ['Logistic ({0})'.format(reg), np.amax(score), score_recall[np.argmax(score)+1],
            score_precision[np.argmax(score)],
            'C = {0}'.format(C[np.argmax(score)]), top_predictor, run_time]


def train_svm(X, y, reg):
    start_time = time.time()
    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.75,
         1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []

    for seed in range(Number_trials):
        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for alpha_run in C:
            if reg == 'l1':
                svc = LinearSVC(C=alpha_run, penalty=reg, loss='squared_hinge', dual=False).fit(
                    X_train, y_train)
            if reg == 'l2':
                svc = LinearSVC(C=alpha_run, penalty=reg).fit(X_train, y_train)
            training_accuracy.append(svc.score(X_train, y_train))
            test_accuracy.append(svc.score(X_test, y_test))
            recall.append(recall_score(y_test, svc.predict(X_test)))
            precision.append(precision_score(y_test, svc.predict(X_test)))
           # if alpha_run == 0.01:
            coefs = svc.coef_
            weighted_coefs.append(coefs)

        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
        recall_test.append(recall)
        precision_test.append(precision)

    # get the mean of the weighted coefficients over all the trials
    mean_coefs = np.mean(weighted_coefs, axis=0)
    score = np.mean(score_test, axis=0)
    score_recall = np.mean(recall_test, axis=0)
    score_precision = np.mean(precision_test, axis=0)

    top_predictor = X.columns[np.argmax(np.abs(mean_coefs))]
    abs_mean_coefs = np.abs(mean_coefs[0, :])
    coefs_count = len(abs_mean_coefs)
    #fig, ax = plt.subplots(figsize=(3, 7))
    #ax.barh(np.arange(coefs_count), sorted(abs_mean_coefs))

    #ax.set_yticks(np.arange(coefs_count))
    #ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])

    run_time = (time.time() - start_time)
    return ['Linear SVM ({0})'.format(reg), np.amax(score),
            score_recall[np.argmax(score)],
            score_precision[np.argmax(score)],
            'C = {0}'.format(C[np.argmax(score)]), top_predictor, run_time]


def train_DT(X, y):
    start_time = time.time()
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []
    maxdepth_settings = list(range(1, 20))

    for seed in range(Number_trials):
        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for depth in maxdepth_settings:
            tree = DecisionTreeClassifier(
                max_depth=depth).fit(X_train, y_train)
            training_accuracy.append(tree.score(X_train, y_train))
            test_accuracy.append(tree.score(X_test, y_test))
            recall.append(recall_score(y_test, tree.predict(X_test)))
            precision.append(precision_score(y_test, tree.predict(X_test)))
            coefs = tree.feature_importances_
            weighted_coefs.append(coefs)

        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
        recall_test.append(recall)
        precision_test.append(precision)

    # get the mean of the weighted coefficients over all the trials
    mean_coefs = np.mean(weighted_coefs, axis=0)
    train_score = np.mean(score_train, axis=0)
    score = np.mean(score_test, axis=0)
    score_recall = np.mean(recall_test, axis=0)
    score_precision = np.mean(precision_test, axis=0)

    top_predictor = X.columns[np.argmax(np.abs(mean_coefs))]
    abs_mean_coefs = np.abs(mean_coefs[:])
    coefs_count = len(abs_mean_coefs)
    #fig, ax = plt.subplots(figsize=(3, 7))
    #ax.barh(np.arange(coefs_count), sorted(abs_mean_coefs))

    #ax.set_yticks(np.arange(coefs_count))
    #ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])

    run_time = (time.time() - start_time)
    return ['Decision Tree',
            np.amax(score), score_recall[np.argmax(score)],
            score_precision[np.argmax(score)],
            'Max depth = {0}'.format(maxdepth_settings[np.argmax(score)]),
            top_predictor,
            run_time]


def train_RF(X, y, reg):
    start_time = time.time()
    max_features_tuning = [0.1, .2, .3, .4, .5, .7, .8]
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []

    for seed in range(Number_trials):
        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed)
        for max_features_run in max_features_tuning:
            svc = RandomForestClassifier(
                n_estimators=100, max_features=max_features_run).fit(X_train, y_train)
            training_accuracy.append(svc.score(X_train, y_train))
            test_accuracy.append(svc.score(X_test, y_test))
            recall.append(recall_score(y_test, svc.predict(X_test)))
            precision.append(precision_score(y_test, svc.predict(X_test)))
           # if alpha_run == 0.01:
            coefs = svc.feature_importances_
            weighted_coefs.append(coefs)

        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
        recall_test.append(recall)
        precision_test.append(precision)

    # get the mean of the weighted coefficients over all the trials
    mean_coefs = np.mean(weighted_coefs, axis=0)
    score = np.mean(score_test, axis=0)
    score_recall = np.mean(recall_test, axis=0)
    score_precision = np.mean(precision_test, axis=0)

    top_predictor = X.columns[np.argmax(np.abs(mean_coefs))]
    abs_mean_coefs = np.abs(mean_coefs[:])
    coefs_count = len(abs_mean_coefs)
    #fig, ax = plt.subplots(figsize=(3, 7))
    #ax.barh(np.arange(coefs_count), sorted(abs_mean_coefs))

    #ax.set_yticks(np.arange(coefs_count))
    #ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])

    run_time = (time.time() - start_time)
    return ['Random Forest', np.amax(score),
            score_recall[np.argmax(score)],
            score_precision[np.argmax(score)],
            'Max_features = {0}'.format(max_features_tuning[np.argmax(score)]), top_predictor, run_time]


def train_GBM(X, y, reg):
    start_time = time.time()
    max_features_tuning = [0.1, .2, .3, .4, .5, .7, .8]
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []

    for seed in range(Number_trials):
        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed)
        for max_features_run in max_features_tuning:
            svc = GradientBoostingClassifier(
                n_estimators=300, max_depth=3, max_features=max_features_run).fit(X_train, y_train)
            training_accuracy.append(svc.score(X_train, y_train))
            test_accuracy.append(svc.score(X_test, y_test))
            recall.append(recall_score(y_test, svc.predict(X_test)))
            precision.append(precision_score(y_test, svc.predict(X_test)))
           # if alpha_run == 0.01:
            coefs = svc.feature_importances_
            weighted_coefs.append(coefs)

        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
        recall_test.append(recall)
        precision_test.append(precision)

    # get the mean of the weighted coefficients over all the trials
    mean_coefs = np.mean(weighted_coefs, axis=0)
    score = np.mean(score_test, axis=0)
    score_recall = np.mean(recall_test, axis=0)
    score_precision = np.mean(precision_test, axis=0)

    top_predictor = X.columns[np.argmax(np.abs(mean_coefs))]
    abs_mean_coefs = np.abs(mean_coefs[:])
    coefs_count = len(abs_mean_coefs)
    #fig, ax = plt.subplots(figsize=(3, 7))
    #ax.barh(np.arange(coefs_count), sorted(abs_mean_coefs))

    #ax.set_yticks(np.arange(coefs_count))
    #ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])
    #ax.set_ylabel('Features')
    #ax.set_xlabel('Importance')

    run_time = (time.time() - start_time)
    return ['Gradient Boosting Method', np.amax(score),
            score_recall[np.argmax(score)],
            score_precision[np.argmax(score)],
            'Max_features = {0}'.format(max_features_tuning[np.argmax(score)]), top_predictor, run_time]

