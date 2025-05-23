import ipaddress
import warnings
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import pickle

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython.core.display_functions import display
from IPython.core.display import HTML
from collections import OrderedDict
from sklearn.decomposition import PCA

temp_dataset_name = "UNSW-NB15-BALANCED-TRAIN.csv"
random_seed = 42

# /Users/anguslin/PycharmProjects/AI_Project_1/NIDS.py
# python3 NIDS.py UNSW-NB15-BALANCED-TRAIN.csv rf label
# python3 /Users/anguslin/PycharmProjects/AI_Project_1/NIDS.py UNSW-NB15-BALANCED-TRAIN.csv rf label

def main():
    forest = "rf"
    knn = "knn"
    mlp = "mlp"

    parser = argparse.ArgumentParser(description='NIDS implementation with ML classifiers trained on UNSW-NB15 dataset.')
    parser.add_argument('-t', '--testset', help='Path of heldout testset csv', required=True)
    parser.add_argument('-c', '--classifier', help=f'Name of classification method: \'{forest}\', \'{knn}\', '
                                                   f'\'{mlp}\')', required=True)
    parser.add_argument('-k', '--task', help='label or attack_cat', required=True)
    parser.add_argument('-m', '--model', help='Name of model to load', required=False)
    args = vars(parser.parse_args())

    if args['model'] is not None:
        X_test, y_test_label, y_test_cat = load_withheld_testset_and_scale(args['testset'])
        model_label_load_predict(model_name=args['classifier'], model=args['model'], feature_cols=X_test, y_labels=y_test_label)
        model_attack_cat_load_predict(model_name=args['classifier'], model=args['model'], feature_cols=X_test, y_labels=y_test_cat)
        return

    classifer = args['classifier']
    task = args['task']

    # model_name = args['model']
    # model = load_modal(model_name)

    X_train, X_val, X_test, y_train_label, y_val_label, y_test_label, y_train_cat, y_val_cat, y_test_cat = \
        get_train_validate_test_sets()

    if classifer == forest:
        if task == "label":
            random_forest_classifier_label(X_train, y_train_label, X_val, y_val_label, X_test, y_test_label)
        elif task == "attack_cat":
            random_forest_classifier_attack_cat(X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat)
    elif classifer == knn:
        if task == "label":
            knn_classifier_label(X_train, y_train_label, X_val, y_val_label, X_test, y_test_label)
        elif task == "attack_cat":
            knn_classifier_attack_cat(X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat)
    elif classifer == mlp:
        if task == "label":
            mlp_classifier_label(X_train, y_train_label, X_val, y_val_label, X_test, y_test_label)
        elif task == "attack_cat":
            mlp_classifier_attack_cat(X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat)


def preprocessing(dataframe):
    # Convert IPv4 addresses to integers
    dataframe['srcip'] = dataframe['srcip'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    dataframe['dstip'] = dataframe['dstip'].apply(lambda x: int(ipaddress.IPv4Address(x)))

    # Convert port values and drop rows with invalid values
    dataframe['sport'] = dataframe['sport'].apply(check_port_int)
    dataframe['dsport'] = dataframe['dsport'].apply(check_port_int)
    dataframe.dropna(subset=['sport'], axis=0, inplace=True)
    dataframe.dropna(subset=['dsport'], axis=0, inplace=True)

    dataframe[['ct_flw_http_mthd', 'is_ftp_login']] = dataframe[['ct_flw_http_mthd', 'is_ftp_login']].fillna(0)
    dataframe['ct_ftp_cmd'] = dataframe['ct_ftp_cmd'].replace(r'^\s+$', np.nan, regex=True).fillna(0)
    # dataframe['ct_ftp_cmd'] = dataframe['ct_ftp_cmd'].astype('int64')
    dataframe['attack_cat'] = dataframe['attack_cat'].astype('string').fillna('None')
    dataframe['attack_cat'] = dataframe['attack_cat'].apply(check_attack_cat)

    feature_cols = dataframe.drop(columns=['Label', 'attack_cat'])

    feature_cols[['proto', 'state', 'service']] = feature_cols[['proto', 'state', 'service']] \
        .apply(lambda x: pd.factorize(x)[0])

    # feature_cols = selected_feature(feature_cols)

    label = dataframe['Label']
    attack_cat = dataframe['attack_cat']

    return feature_cols, label, attack_cat


def selected_feature(dataframe):
    feature_cols = dataframe[['srcip', 'dsport', 'service', 'dbytes', 'sbytes']]
    return feature_cols


def create_train_val_test_set():
    unsw1_data = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv")

    train_set, temp_set = train_test_split(unsw1_data, test_size=0.3, random_state=random_seed)
    val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=random_seed)

    train_set.to_csv('train.csv', index=False)
    val_set.to_csv('validate.csv', index=False)
    test_set.to_csv('test.csv', index=False)


def load_withheld_testset_and_scale(testset):
    # for withheld testset, use with loaded models
    test = pd.read_csv(testset)
    X_test, y_test_label, y_test_cat = preprocessing(test)

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    return X_test, y_test_label, y_test_cat


def get_train_validate_test_sets():
    train = pd.read_csv('train.csv')
    val = pd.read_csv('validate.csv')
    test = pd.read_csv('test.csv')

    X_train, y_train_label, y_train_cat = preprocessing(train)
    X_val, y_val_label, y_val_cat = preprocessing(val)
    X_test, y_test_label, y_test_cat = preprocessing(test)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train_label, y_val_label, y_test_label, y_train_cat, y_val_cat, y_test_cat


def load_modal(model):
    plk = open(model, "rb")
    model = pickle.load(plk)
    plk.close()
    return model


def model_label_load_predict(model_name, model, feature_cols, y_labels):
    clf = load_modal(model)
    y_pred = clf.predict(feature_cols)

    print(f"\n{model_name}, Label (Test Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_labels, y_pred) * 100))
    print(metrics.classification_report(y_labels, y_pred, digits=4))


def model_attack_cat_load_predict(model_name, model, feature_cols, y_labels):
    clf = load_modal(model)
    y_pred = clf.predict(feature_cols)

    print(f"\n{model_name}, Attack Category (Test Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_labels, y_pred) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
                      'Analysis', 'Shellcode', 'Worms']
    print(metrics.classification_report(y_labels, y_pred, labels=attack_columns, digits=4))


def random_forest_classifier_label(X_train, y_train, X_val, y_val, X_test, y_test):

    clf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=random_seed, n_jobs=-1)
    # clf = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=random_seed, max_features=None, n_jobs=-1)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\nRandom Forest, Label, (Validation Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_val, y_pred=y_pred) * 100))
    print(metrics.classification_report(y_true=y_val, y_pred=y_pred, digits=4))

    y_pred = clf.predict(X_test)
    print("\nRandom Forest, Label, (Test Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_test, y_pred=y_pred) * 100))
    print(metrics.classification_report(y_true=y_test, y_pred=y_pred, digits=4))

    pickle.dump(clf, open('rf_label_model.pickle', 'wb'))

    # feature_col = X_train.columns.to_list()
    # random_forest_graph(clf, feature_col, "Random Forest Feature Importance (Label)")



def random_forest_classifier_attack_cat(X_train, y_train, X_val, y_val, X_test, y_test):

    # clf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=random_seed, n_jobs=-1)
    clf = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=random_seed, max_features=None, n_jobs=-1)
    # clf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=random_seed, min_samples_split=20, max_depth=140, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(f"Random Forest, Attack Category (Validation Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_val, y_pred) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
                      'Analysis', 'Shellcode', 'Worms']
    print(metrics.classification_report(y_val, y_pred, labels=attack_columns, digits=4))

    y_pred = clf.predict(X_test)
    print(f"\nRandom Forest, Attack Category (Test Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
                      'Analysis', 'Shellcode', 'Worms']
    print(metrics.classification_report(y_test, y_pred, labels=attack_columns, digits=4))

    pickle.dump(clf, open('rf_attack_cat_model.pickle', 'wb'))

    # feature_col = X_train.columns.to_list()
    # random_forest_graph(clf, feature_col, "Random Forest Feature Importance (Attack Category)")



def knn_classifier_label(X_train, y_train, X_val, y_val, X_test, y_test):

    clf = KNeighborsClassifier(n_neighbors=6, metric='manhattan', p=1, weights='distance')
    clf.fit(X_train, y_train)

    # Grid search for hyper parameter tuning
    """ grid_params = {'n_neighbors': [29], 'weights': ['uniform', 'distance'], 'metric': ['minkowski','euclidean','manhattan']}
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=3, cv=3, n_jobs=-1, scoring='f1_macro')
    g_res = gs.fit(X_train, y_train)
    print(g_res.best_score_)
    print(g_res.best_params_) 
    print(g_res.cv_results_) """

    y_pred = clf.predict(X_val)
    print("\nKNN, Label, (Validation Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_val, y_pred=y_pred) * 100))
    print(metrics.classification_report(y_true=y_val, y_pred=y_pred, digits=4))

    y_pred = clf.predict(X_test)
    print("\nKNN, Label, (Test Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_test, y_pred=y_pred) * 100))
    print(metrics.classification_report(y_true=y_test, y_pred=y_pred, digits=4))

    pickle.dump(clf, open('knn_label_model.pickle', 'wb'))



def knn_classifier_attack_cat(X_train, y_train, X_val, y_val, X_test, y_test):

    X_train, X_val, X_test = Recursive_Feature_Elim_CV(X_train, y_train, X_val, X_test)
    # finding optimal k using macro F1 scores
    """ acc = []
    for i in range (1,40):
        print("Training K value:",i)
        clf = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc.append(metrics.f1_score(y_test,y_pred, average='macro')) """

    # plot macro-F1 score vs. K value
    """ plt.figure(figsize=(10,6))
    plt.plot(range(1,20),acc,color = 'blue', linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
    plt.title('F1 Score vs K value')
    plt.xlabel('K')
    plt.ylabel('F1 Score')
    plt.show()
    print("Maximum Accuracy:-",max(acc), "at K =",acc.index(max(acc))) """

    # Grid search for hyper parameter tuning
    # grid_params = {'n_neighbors': [6], 'weights': ['uniform', 'distance'], 'metric': ['euclidean','manhattan']}
    # gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=3, cv=2, n_jobs=-1, scoring='f1_macro')
    # g_res = gs.fit(X_train, y_train)
    # print(g_res.best_score_)
    # print(g_res.best_params_) 
    # print(g_res.cv_results_)

    n_neighbors = 6
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='manhattan', p=1, weights='distance')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    print(f"KNN, Attack Category (Validation Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_val, y_pred) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
                      'Analysis', 'Shellcode', 'Worms']
    print(attack_columns)
    print(y_val.shape)
    print(y_val.head())
    
    print(metrics.classification_report(y_val, y_pred, labels=attack_columns, digits=4))
    
    # y_pred = clf.predict(X_test)
    # print(f"\nKNN, Attack Category (Test Set) \n---------------------------")
    # print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    # attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
    #                   'Analysis', 'Shellcode', 'Worms']
    # print(metrics.classification_report(y_test, y_pred, labels=attack_columns, digits=4))
    
    # pickle.dump(clf, open('knn_attack_cat_model.pickle', 'wb'))



def mlp_classifier_label(X_train, y_train, X_val, y_val, X_test, y_test):

    clf = MLPClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\nMLP, Label, (Validation Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_val, y_pred=y_pred) * 100))
    print(metrics.classification_report(y_true=y_val, y_pred=y_pred, digits=4))

    y_pred = clf.predict(X_test)
    print("\nMLP, Label, (Test Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_test, y_pred=y_pred) * 100))
    print(metrics.classification_report(y_true=y_test, y_pred=y_pred, digits=4))

    pickle.dump(clf, open('mlp_label_model.pickle', 'wb'))

    # feature_col = X_train.columns.to_list()
    # random_forest_graph(clf, feature_col, "Random Forest Feature Importance (Label)")



def mlp_classifier_attack_cat(X_train, y_train, X_val, y_val, X_test, y_test):

    clf = MLPClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(f"MLP, Attack Category (Validation Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_val, y_pred) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
                      'Analysis', 'Shellcode', 'Worms']
    print(metrics.classification_report(y_val, y_pred, labels=attack_columns, digits=4))

    y_pred = clf.predict(X_test)
    print(f"\nMLP, Attack Category (Test Set) \n---------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
                      'Analysis', 'Shellcode', 'Worms']
    print(metrics.classification_report(y_test, y_pred, labels=attack_columns, digits=4))

    pickle.dump(clf, open('mlp_attack_cat_model.pickle', 'wb'))

    # feature_col = X_train.columns.to_list()
    # random_forest_graph(clf, feature_col, "Random Forest Feature Importance (Attack Category)")



def RFE_model_eval(X_train, y_train):

    # get a list of models to evaluate
    def get_models():
        models = dict()
        # perceptron
        rfe = RFE(estimator=Perceptron(), n_features_to_select=5,step=3)
        model = DecisionTreeClassifier()
        models['Perceptron'] = Pipeline(steps=[('s',rfe),('m',model)])
        # cart
        rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5,step=3)
        model = DecisionTreeClassifier()
        models['DecisionTree'] = Pipeline(steps=[('s',rfe),('m',model)])
        # rf
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50), n_features_to_select=5,step=3)
        model = DecisionTreeClassifier()
        models['RandomForest'] = Pipeline(steps=[('s',rfe),('m',model)])
        return models

    def evaluate_model(model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores
    
    # define dataset
    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X_train, y_train)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()

def Recursive_Feature_Elim(X_train, y_train, X_val, X_test):
    # Perform RFE, selecting 5 features
    rfe = RFE(estimator=DecisionTreeClassifier(), step=2,n_jobs=-1, n_features_to_select=5)
    rfe.fit(X_train, y_train)
    print(X_train.columns[(rfe.get_support())])
    X_train = rfe.transform(X_train)
    X_val = rfe.transform(X_val)   
    X_test = rfe.transform(X_test)

    return X_train, X_val, X_test 

def Recursive_Feature_Elim_CV(X_train, y_train, X_val, X_test):
    # Perform RFE with cross validation for automatic selection of the best features
    rfe = RFECV(estimator=DecisionTreeClassifier(), step=2,n_jobs=-1)
    rfe.fit(X_train, y_train)
    X_train = rfe.transform(X_train)
    X_val = rfe.transform(X_val)
    X_test = rfe.transform(X_test)

    # Plot RFECV feature # vs. mean accuracy graph

    # n_scores = len(rfe.cv_results_["mean_test_score"])
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Mean test accuracy")
    # plt.errorbar(
    #     range(min_features_to_select, n_scores + min_features_to_select),
    #     rfe.cv_results_["mean_test_score"],
    #     yerr=rfe.cv_results_["std_test_score"],
    # )
    # plt.title("Recursive Feature Elimination \nwith correlated features")
    # plt.show()

    # rfe_df = pd.DataFrame({'Features': feature_cols.columns, 'Kept': rfe.support_, 'Rank': rfe.ranking_})
    # rfe_df.to_csv('rfe_rank.csv')

    return X_train, X_val, X_test

def random_forest_graph(clf, feature_col, title):
    plt.figure(figsize=(20, 15))
    importance = clf.feature_importances_
    idxs = np.argsort(importance)
    plt.title(title)
    plt.barh(range(len(idxs)), importance[idxs], align="center")
    plt.yticks(range(len(idxs)), [feature_col[i] for i in idxs])
    plt.xlabel("Random Forest Feature Importance")
    # plt.tight_layout()
    plt.show()

    # Get numerical feature importances
    importances = list(clf.feature_importances_)  # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                           zip(feature_col, importances)]  # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)  # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]



def random_forest_OOB_error_rate(X_train, y_train, y_train_cat):
    ensemble_clfs = [
        (
            "RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=random_seed,
            ),
        ),
        (
            "RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                warm_start=True,
                max_features="log2",
                oob_score=True,
                random_state=random_seed,
            ),
        ),
        (
            "RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=random_seed,
            ),
        ),
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 10
    max_estimators = 420

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1, 20):
            print(f"current_progress: {i}")
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()

    ensemble_clfs_cat = [
        (
            "RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=random_seed,
            ),
        ),
        (
            "RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                warm_start=True,
                max_features="log2",
                oob_score=True,
                random_state=random_seed,
            ),
        ),
        (
            "RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=random_seed,
            ),
        ),
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs_cat)

    # Range of `n_estimators` values to explore.
    min_estimators = 15
    max_estimators = 200

    for label, clf in ensemble_clfs_cat:
        for i in range(min_estimators, max_estimators + 1, 5):
            print(f"current_progress: {i}")
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train_cat)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.title("Forest OOB Error vs Forest Size (Atk Category)")
    plt.show()


# convert hexadecimal port values to integer
def check_port_int(value):
    try:
        return int(str(value), 0)
    except ValueError:
        return np.NaN


### Ask teacher is need to handle typos in attack cat types (like "Backdoor" and "Backdoors", or "Fuzzers" and " Fuzzers ")
def check_attack_cat(value):
    try:
        x = value.strip()
        # if x == "Backdoor":
        #     x = "Backdoors"
        return x
    except ValueError:
        return np.NaN


def feature_analysis_PCA(X_train, y_train, y_train_cat, X_val, y_val, y_val_cat):
    # X_train = pd.concat([X_train, y_train], axis=1)
    # subsample = X_train.groupby('Label').sample(frac=0.001)
    # X_train = subsample.drop(columns=['Label'])
    # y_train = subsample['Label']

    # Create color_label column
    df_temp = pd.concat([X_train, y_train], axis=1)
    df_temp['color_label'] = df_temp.apply(lambda row: label_color(row), axis=1)
    y_color = df_temp['color_label']
    # Create color_atkcat column
    df_temp_cat = pd.concat([X_train, y_train_cat], axis=1)
    df_temp_cat['color_atkcat'] = df_temp_cat.apply(lambda row: atk_cat_color(row), axis=1)
    y_color_cat = df_temp_cat['color_atkcat']

    # # # Define which columns should be encoded vs scaled
    # columns_to_encode = ['proto', 'state', 'service']
    # columns_to_scale = X_train.drop(columns=['proto', 'state', 'service']).columns
    #
    # # Instantiate encoder/scaler
    # scaler = StandardScaler()
    # ohe = OneHotEncoder(sparse=False)
    #
    # # Scale and Encode Separate Columns
    # encoded_columns = ohe.fit_transform(X_train[columns_to_encode])
    # scaled_columns = scaler.fit_transform(X_train[columns_to_scale])
    # encoded_pd = pd.DataFrame(encoded_columns, columns=ohe.get_feature_names_out(['proto', 'state', 'service']))
    # scaled_pd = pd.DataFrame(scaled_columns, columns=columns_to_scale)
    #
    # # Concatenate (Column-Bind) Processed Columns Back Together
    # X_train = pd.concat([scaled_pd, encoded_pd], axis=1)
    #
    # # Do the same for X_val (and maybe X_test later)
    # encoded_columns_val = ohe.transform(X_val[columns_to_encode])
    # scaled_columns_val = scaler.transform(X_val[columns_to_scale])
    # encoded_pd_val = pd.DataFrame(encoded_columns_val, columns=ohe.get_feature_names_out(['proto', 'state', 'service']))
    # scaled_pd_val = pd.DataFrame(scaled_columns_val, columns=columns_to_scale)
    # X_val = pd.concat([scaled_pd_val, encoded_pd_val], axis=1)


    scaler = StandardScaler()
    scaled_columns = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(scaled_columns, columns=X_train.columns)


    pca = PCA(n_components=None)
    X_train_pca = pca.fit_transform(X_train)
    feature_len = len(X_train.columns)


    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.cumsum())
    print(len(np.cumsum(pca.explained_variance_ratio_)))
    print(len(pca.explained_variance_ratio_.cumsum()))

    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, feature_len+1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, feature_len, step=10))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('Number of PCA components to capture total variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.show()

    pca_index = []
    for i in range(1, len(X_train.columns)+1):
        pca_index.append(f"PCA{i}")

    pca_components = abs(pca.components_)
    print(pd.DataFrame(pca_components, columns=X_train.columns, index=pca_index))

    for row in range(pca_components.shape[0]):
        # get the indices of the top 4 values in each row
        temp = np.argpartition(-(pca_components[row]), 4)

        # sort the indices in descending order
        indices = temp[np.argsort((-pca_components[row])[temp])][:4]

        # print the top 4 feature names
        print(f'Component {row}: {X_train.columns[indices].to_list()}')

    # pca = PCA(n_components=0.95)
    # # pca.fit(X_train)
    # # reduced = pca.transform(X_train)
    # X_train_pca = pca.fit_transform(X_train)

    # print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.cumsum())
    print(len(pca.explained_variance_ratio_.cumsum()))


    biplot(X_train_pca[:, 0:2], np.transpose(pca.components_[0:2, :]), y_color, "1", "2", labels=X_train.columns)
    plt.show()

    biplot(X_train_pca[:, 2:4], np.transpose(pca.components_[2:4, :]), y_color, "3", "4", labels=X_train.columns)
    plt.show()

    biplot(X_train_pca[:, 4:6], np.transpose(pca.components_[4:6, :]), y_color, "5", "6", labels=X_train.columns)
    plt.show()

    biplot(X_train_pca[:, 0:2], np.transpose(pca.components_[0:2, :]), y_color_cat, "1", "2", labels=X_train.columns)
    plt.show()

    biplot(X_train_pca[:, 2:4], np.transpose(pca.components_[2:4, :]), y_color_cat, "3", "4", labels=X_train.columns)
    plt.show()

    biplot(X_train_pca[:, 4:6], np.transpose(pca.components_[4:6, :]), y_color_cat, "5", "6", labels=X_train.columns)
    plt.show()


    X_val_pca = pca.transform(X_val)


    return X_train_pca, y_train, y_train_cat, X_val_pca, y_val, y_val_cat


def biplot(score,coeff, y, pca_val1, pca_val2, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c= y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pca_val1))
    plt.ylabel("PC{}".format(pca_val2))
    plt.grid()

def label_color (row):
   if row['Label'] == 0 :
      return 'Green'
   if row['Label'] == 1 :
      return 'Red'
   return 'Black'

def atk_cat_color (row):
    labels = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor', 'Analysis', 'Shellcode', 'Worms']
    if row['attack_cat'] == 'None':
        return 'green'
    if row['attack_cat'] == 'Generic':
        return 'red'
    if row['attack_cat'] == 'Fuzzers':
        return 'blue'
    if row['attack_cat'] == 'Exploits':
        return 'orange'
    if row['attack_cat'] == 'DoS':
        return 'black'
    if row['attack_cat'] == 'Reconnaissance':
        return 'cyan'
    if row['attack_cat'] == 'Backdoor' or row['attack_cat'] == 'Backdoors':
        return 'brown'
    if row['attack_cat'] == 'Analysis':
        return 'pink'
    if row['attack_cat'] == 'Shellcode':
        return 'yellow'
    if row['attack_cat'] == 'Worms':
        return 'magenta'
    return 'green'

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("END")
    print(f"--- {(time.time() - start_time):.2f} seconds ---")
