import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

import warnings
warnings.filterwarnings("ignore")


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
    skf = StratifiedKFold(n_splits=5)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

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
            acc_test.append(clf.score(X_val, y_val))
            recall.append(recall_score(y_val, clf.predict(X_val)))
            precision.append(precision_score(y_val, clf.predict(X_val)))

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
            np.amax(score),
            score_recall[np.argmax(score)+1],
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
    skf = StratifiedKFold(n_splits=5)

    for seed in range(Number_trials):
        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []
        
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            for alpha_run in C:
                if reg == 'l1':
                    lr = LogisticRegression(
                        C=alpha_run, penalty=reg, solver='liblinear').fit(X_train, y_train)
                if reg == 'l2':
                    lr = LogisticRegression(
                        C=alpha_run, penalty=reg, dual=False).fit(X_train, y_train)

                training_accuracy.append(lr.score(X_train, y_train))
                test_accuracy.append(lr.score(X_val, y_val))
                recall.append(recall_score(y_val, lr.predict(X_val)))
                precision.append(precision_score(y_val, lr.predict(X_val)))
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
    return ['Logistic ({0})'.format(reg),
            np.amax(score),
            score_recall[np.argmax(score)+1],
            score_precision[np.argmax(score)],
            'C = {0}'.format(C[np.argmax(score)]),
            top_predictor,
            run_time]


def train_svm(X, y, reg):
    start_time = time.time()
    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.75,
         1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []
    skf = StratifiedKFold(n_splits=5)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        for alpha_run in C:
            if reg == 'l1':
                svc = LinearSVC(C=alpha_run, penalty=reg, loss='squared_hinge', dual=False).fit(
                    X_train, y_train)
            if reg == 'l2':
                svc = LinearSVC(C=alpha_run, penalty=reg).fit(X_train, y_train)
            training_accuracy.append(svc.score(X_train, y_train))
            test_accuracy.append(svc.score(X_val, y_val))
            recall.append(recall_score(y_val, svc.predict(X_val)))
            precision.append(precision_score(y_val, svc.predict(X_val)))
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


def train_DT(X, y, resample='none', sampling_strategy=0.5):
    start_time = time.time()
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []
    maxdepth_settings = list(range(1, 20))
    skf = StratifiedKFold(n_splits=5)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        if resample == 'SMOTE':
            X_train, y_train = SMOTE(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
        elif resample == 'ADASYN':
            X_train, y_train = ADASYN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
        elif resample == 'SMOTENC':
            X_train, y_train = SMOTENC(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143, 
                categorical_features=[1, 3, 4, 5, 18, 19]).fit_resample(X_train, y_train)
            
        elif resample == 'SMOTEENN':
            X_train, y_train = SMOTEENN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143,
                smote=SMOTE(
                    sampling_strategy=sampling_strategy,
                    n_jobs=-1,
                    random_state=143),
                enn=EditedNearestNeighbours(
                    sampling_strategy='all',
                    n_jobs=-1)
                ).fit_resample(X_train, y_train)
            
        else:
            pass
            
        for depth in maxdepth_settings:
            tree = DecisionTreeClassifier(
                max_depth=depth).fit(X_train, y_train)
            training_accuracy.append(tree.score(X_train, y_train))
            test_accuracy.append(tree.score(X_val, y_val))
            recall.append(recall_score(y_val, tree.predict(X_val)))
            precision.append(precision_score(y_val, tree.predict(X_val)))
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


def train_RF(X, y, reg, resample='none', sampling_strategy=0.5):
    start_time = time.time()
    max_features_tuning = [0.1, .2, .3, .4, .5, .7, .8]
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []
    skf = StratifiedKFold(n_splits=5)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []
        
        if resample == 'SMOTE':
            X_train, y_train = SMOTE(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
        elif resample == 'ADASYN':
            X_train, y_train = ADASYN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
        elif resample == 'SMOTENC':
            X_train, y_train = SMOTENC(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143, 
                categorical_features=[1, 3, 4, 5, 18, 19]).fit_resample(X_train, y_train)
        
        elif resample == 'SMOTEENN':
            X_train, y_train = SMOTEENN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143,
                smote=SMOTE(
                    sampling_strategy=sampling_strategy,
                    n_jobs=-1,
                    random_state=143),
                enn=EditedNearestNeighbours(
                    sampling_strategy='all',
                    n_jobs=-1)
                ).fit_resample(X_train, y_train)
            
        else:
            pass

        for max_features_run in max_features_tuning:
            svc = RandomForestClassifier(
                n_estimators=100, max_features=max_features_run).fit(X_train, y_train)
            training_accuracy.append(svc.score(X_train, y_train))
            test_accuracy.append(svc.score(X_val, y_val))
            recall.append(recall_score(y_val, svc.predict(X_val)))
            precision.append(precision_score(y_val, svc.predict(X_val)))
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


def train_GBM(X, y, reg, resample='none', sampling_strategy=0.5):
    start_time = time.time()
    max_features_tuning = [0.1, .2, .3, .4, .5, .7, .8]
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []
    skf = StratifiedKFold(n_splits=5)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []
        
        if resample == 'SMOTE':
            X_train, y_train = SMOTE(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
        elif resample == 'ADASYN':
            X_train, y_train = ADASYN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
        elif resample == 'SMOTENC':
            X_train, y_train = SMOTENC(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143, 
                categorical_features=[1, 3, 4, 5, 18, 19]).fit_resample(X_train, y_train)
            
        elif resample == 'SMOTEENN':
            X_train, y_train = SMOTEENN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143,
                smote=SMOTE(
                    sampling_strategy=sampling_strategy,
                    n_jobs=-1,
                    random_state=143),
                enn=EditedNearestNeighbours(
                    sampling_strategy='all',
                    n_jobs=-1)
                ).fit_resample(X_train, y_train)
            
        else:
            pass

        for max_features_run in max_features_tuning:
            svc = GradientBoostingClassifier(
                n_estimators=300, max_depth=3, max_features=max_features_run).fit(X_train, y_train)
            training_accuracy.append(svc.score(X_train, y_train))
            test_accuracy.append(svc.score(X_val, y_val))
            recall.append(recall_score(y_val, svc.predict(X_val)))
            precision.append(precision_score(y_val, svc.predict(X_val)))
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


def train_ABC(X, y, reg, resample='none', sampling_strategy=0.5):
    start_time = time.time()
    score_train = []
    score_test = []
    recall_test = []
    precision_test = []
    weighted_coefs = []
    skf = StratifiedKFold(n_splits=5)
    n_estimators = [50, 100, 150, 200, 250, 300, 350, 400]

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        training_accuracy = []
        test_accuracy = []
        recall = []
        precision = []
        
        if resample == 'SMOTE':
            X_train, y_train = SMOTE(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
        elif resample == 'ADASYN':
            X_train, y_train = ADASYN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
        elif resample == 'SMOTENC':
            X_train, y_train = SMOTENC(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143, 
                categorical_features=[1, 3, 4, 5, 18, 19]).fit_resample(X_train, y_train)
            
        elif resample == 'SMOTEENN':
            X_train, y_train = SMOTEENN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143,
                smote=SMOTE(
                    sampling_strategy=sampling_strategy,
                    n_jobs=-1,
                    random_state=143),
                enn=EditedNearestNeighbours(
                    sampling_strategy='all',
                    n_jobs=-1)
                ).fit_resample(X_train, y_train)
            
        else:
            pass
        
        for estimators in n_estimators:
            svc = AdaBoostClassifier(
                    n_estimators=estimators, random_state=143).fit(X_train, y_train)
            training_accuracy.append(svc.score(X_train, y_train))
            test_accuracy.append(svc.score(X_val, y_val))
            recall.append(recall_score(y_val, svc.predict(X_val)))
            precision.append(precision_score(y_val, svc.predict(X_val)))
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
    return ['AdaBoost Classifier', np.amax(score),
            score_recall[np.argmax(score)],
            score_precision[np.argmax(score)],
            'n_estimators = {0}'.format(n_estimators[np.argmax(score)]), top_predictor, run_time]


def conf_matrix(X, y, fig_num, resample='none', sampling_strategy=0.5):
    skf = StratifiedKFold(n_splits=5)
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if resample == 'SMOTE':
            X_train, y_train = SMOTE(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
            models_dict = {
                'Decision Tree': DecisionTreeClassifier(
                    max_depth=9).fit(X_train, y_train),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_features=0.4).fit(X_train, y_train),
                'GradientBoostingClassifier': GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=3,
                    max_features=0.3).fit(X_train, y_train)  
            }
            
        elif resample == 'ADASYN':
            X_train, y_train = ADASYN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143).fit_resample(X_train, y_train)
            
            models_dict = {
                'Decision Tree': DecisionTreeClassifier(
                    max_depth=8).fit(X_train, y_train),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_features=0.4).fit(X_train, y_train),
                'GradientBoostingClassifier': GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=3,
                    max_features=0.3).fit(X_train, y_train)  
            }
            
        elif resample == 'SMOTENC':
            X_train, y_train = SMOTENC(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143, 
                categorical_features=[1, 2, 3, 4, 5, 18, 19]).fit_resample(X_train, y_train)
            
            models_dict = {
                'Decision Tree': DecisionTreeClassifier(
                    max_depth=10).fit(X_train, y_train),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_features=0.4).fit(X_train, y_train),
                'GradientBoostingClassifier': GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=3,
                    max_features=0.2).fit(X_train, y_train) 
            }
            
        elif resample == 'SMOTEENN':
            X_train, y_train = SMOTEENN(
                sampling_strategy=sampling_strategy,
                n_jobs=-1,
                random_state=143,
                smote=SMOTE(
                    sampling_strategy=sampling_strategy,
                    n_jobs=-1,
                    random_state=143),
                enn=EditedNearestNeighbours(
                    sampling_strategy='all',
                    n_jobs=-1)
                ).fit_resample(X_train, y_train)
            
            models_dict = {
                'Decision Tree': DecisionTreeClassifier(
                    max_depth=10).fit(X_train, y_train),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_features=0.4).fit(X_train, y_train),
                'GradientBoostingClassifier': GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=3,
                    max_features=0.2).fit(X_train, y_train) 
            }   
            
        else:
            models_dict = {
                'Decision Tree': DecisionTreeClassifier(
                    max_depth=11).fit(X_train, y_train),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_features=0.5).fit(X_train, y_train),
                'GradientBoostingClassifier': GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=3,
                    max_features=0.7).fit(X_train, y_train)  
            }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    index = 0
    for model_name, model in models_dict.items():
        plot_confusion_matrix(model, X_val, y_val,
                          ax=axes[index], cmap='summer')
        axes[index].set_title(model_name, fontsize=12)
        axes[index].set_xlabel('Predicted Label', fontsize=12)
        axes[index].set_ylabel('True Label', fontsize=12)
        index += 1
        
    if resample == 'none':
        fig.suptitle(f'Figure {fig_num}. Confusion Matrix without Resampling')
    else:
        fig.suptitle(f'Figure {fig_num}. Confusion Matrix using {resample}')