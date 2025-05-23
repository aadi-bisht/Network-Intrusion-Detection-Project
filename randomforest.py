import ipaddress
import warnings

from sklearn.feature_selection import RFECV

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold, GridSearchCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from graphviz import Source
from sklearn import tree
import time
from collections import OrderedDict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn import metrics # is used to create classification results
from sklearn.tree import export_graphviz # is used for plotting the decision tree
from six import StringIO # is used for plotting the decision tree
from IPython.display import Image # is used for plotting the decision tree
from IPython.core.display_functions import display
from IPython.core.display import HTML # is used for showing the confusion matrix
import pydotplus # is used for plotting the decision tree

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
import pickle


# Maybe move read data to a separate file for all analysis technique files to call
random_seed = 42# pd.set_option('display.max_columns', None)


def read_data(csv_file):
    #
    # train_set, temp_set = train_test_split(unsw_data, test_size=0.3, random_state=random_seed)
    # val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=random_seed)
    #
    # print(train_set.shape)
    # print(val_set.shape)
    # print(test_set.shape)
    # train_set.to_csv('train.csv')
    # val_set.to_csv('validate.csv')
    # test_set.to_csv('test.csv')
    #
    # return

    # unsw_data = pd.read_csv("train.csv")
    # unsw_data = pd.read_csv("validate.csv")
    # unsw_data = pd.read_csv("test.csv")

    unsw_data = pd.read_csv(csv_file)


    # Convert IPv4 addresses to integers
    unsw_data['srcip'] = unsw_data['srcip'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    unsw_data['dstip'] = unsw_data['dstip'].apply(lambda x: int(ipaddress.IPv4Address(x)))

    # Convert port values and drop rows with invalid values
    unsw_data['sport'] = unsw_data['sport'].apply(check_port_int)
    unsw_data['dsport'] = unsw_data['dsport'].apply(check_port_int)
    unsw_data.dropna(subset=['sport'], axis=0, inplace=True)
    unsw_data.dropna(subset=['dsport'], axis=0, inplace=True)
    # feature_cols['sport'] = feature_cols['sport'].apply(lambda x: int(str(x), 0))
    # feature_cols['dsport'] = feature_cols['dsport'].apply(lambda x: int(str(x), 0))
    # print(feature_cols['sport'].apply(check_port_int).dropna())
    # print(feature_cols['dsport'].apply(check_port_int).dropna())

    # Handle missing values and white space for numeric and categorical columns.
    # numeric_columns = feature_cols.select_dtypes(include=np.number).columns
    # # feature_cols[numeric_columns] = feature_cols[numeric_columns].fillna(feature_cols.mean())
    # 0.5 of an ftp_login attempt does not make sense, so 0.
    unsw_data[['ct_flw_http_mthd', 'is_ftp_login']] = unsw_data[['ct_flw_http_mthd', 'is_ftp_login']].fillna(0)
    unsw_data['ct_ftp_cmd'] = unsw_data['ct_ftp_cmd'].replace(r'^\s+$', np.nan, regex=True).fillna(0)
    unsw_data['ct_ftp_cmd'] = unsw_data['ct_ftp_cmd'].astype('int64')
    unsw_data['attack_cat'] = unsw_data['attack_cat'].astype('string').fillna('None')
    unsw_data['attack_cat'] = unsw_data['attack_cat'].apply(check_attack_cat) #.factorize()[0]

    # code, unique = unsw_data['attack_cat'].apply(check_attack_cat).factorize()
    # unsw_data.to_csv('fillna_UNSW-NB15-BALANCED-TRAIN.csv', index=False)
    # print(unsw_data.isna().sum())  # Check all missing values in dataset are handled
    # print(unsw_data.dtypes)  # Verify all datatypes converted to numerical


    y_label = unsw_data[['Label', 'attack_cat']]
    feature_cols = unsw_data.drop(columns=['Label', 'attack_cat'])

    # (Graph thingy)
    # feature_analysis(unsw_data)

    # Factorize categorical data
    # feature_cols[['proto', 'state']] = feature_cols[['proto', 'state']] \
    #     .apply(lambda x: pd.factorize(x)[0])
    feature_cols[['proto', 'state', 'service']] = feature_cols[['proto', 'state', 'service']] \
        .apply(lambda x: pd.factorize(x)[0])

    # feature_cols = feature_cols.drop(columns=['srcip', 'dstip', 'service'])
    # feature_cols = feature_cols.drop(columns=['srcip', 'dstip', 'service', 'proto', 'state'])
    # feature_cols = feature_cols.drop(columns=['srcip', 'dstip', 'service', 'is_ftp_login', 'ct_ftp_cmd', 'is_sm_ips_ports',
    #                                           'dwin', 'trans_depth'])

    # feature_cols = feature_cols[['sttl', 'ct_srv_dst', 'sbytes', 'smeansz', 'proto', 'ct_state_ttl', 'sloss',
    #                              'synack', 'ct_dst_src_ltm', 'dmeansz', 'ct_srv_src', 'service', 'ct_dst_sport_ltm',
    #                              'dbytes', 'dloss', 'state', 'tcprtt', 'ct_src_dport_ltm', 'dbytes']]

    # feature_cols = feature_cols[['srcip', 'dsport', 'service', 'dbytes', 'sbytes']]
    feature_cols  = feature_cols[['ct_dst_src_ltm', 'tcprtt', 'Dpkts', 'sbytes', 'Sintpkt', 'is_ftp_login',
            'Djit', 'Stime', 'Sjit', 'Stime', 'smeansz', 'dsport', 'dur', 'is_sm_ips_ports', 'sport',
            'Sload', 'proto', 'res_bdy_len', 'ct_flw_http_mthd', 'Djit', 'Dload', 'dstip']]

    X_train, X_temp, y_train, y_temp = train_test_split(feature_cols, y_label, test_size=0.1, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_seed)

    # X_temp, X_test, y_temp, y_test = train_test_split(feature_cols, y_label, test_size=0.3, random_state=random_seed)
    # X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_seed)


    y_train_label = y_train.drop(columns=['attack_cat'])
    y_train_cat = y_train.drop(columns=['Label'])
    y_val_label = y_val.drop(columns=['attack_cat'])
    y_val_cat = y_val.drop(columns=['Label'])
    y_test_label = y_test.drop(columns=['attack_cat'])
    y_test_cat = y_test.drop(columns=['Label'])

    # random_forest_OOB_error_rate(X_train, y_train_label, y_train_cat, X_val, y_val_label, y_val_cat,
    #                                X_test, y_test_label, y_test_cat)

    # (PCA Graph)
    # X_train, y_train_label, y_train_cat, X_val, y_val_label, y_val_cat = \
    #     feature_analysis_PCA(X_train, y_train_label, y_train_cat, X_val, y_val_label, y_val_cat)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    random_forest_classifier_label(X_train, y_train_label, y_train_cat, X_val, y_val_label, y_val_cat,
                                   X_test, y_test_label, y_test_cat)
    random_forest_classifier_atk_cat(X_train, y_train_label, y_train_cat, X_val, y_val_label, y_val_cat,
                                     X_test, y_test_label, y_test_cat)
    # random_forest_classifier_RFECV(X_train, y_train_label, y_train_cat, X_val, y_val_label, y_val_cat)
    # random_forest_randomizedSearchCV(X_train, y_train_label, y_train_cat, X_val, y_val_label, y_val_cat)

    # X_train = normalization(X_train)
    # X_val = normalization(X_val)
    # X_test = mean_normalization(X_test) # Not used in training

    return X_train, y_train_label, y_train_cat, X_val, y_val_label, y_val_cat, X_test, y_test_label, y_test_cat


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


def feature_analysis(df):
    service_df = df['service'].value_counts(normalize=True) * 100
    proto_df = df['proto'].value_counts(normalize=True) * 100
    state_df = df['state'].value_counts(normalize=True) * 100

    print("\'Service\' value distribution")
    print(service_df)
    print("\'proto\' value distribution")
    print(proto_df)
    print("\'state\' value distribution")
    print(state_df)

    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    service_val = service_df.index
    service_count = service_df.values
    plt.bar(service_val, service_count, color='maroon', width=0.4)
    plt.xlabel("Unique values in \'Service\'")
    plt.ylabel("Percentage of value in column")
    plt.title("\'Service\' Value Distribution")
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    proto_val = proto_df.index[:14]
    proto_count = proto_df.values[:14]
    plt.bar(proto_val, proto_count, color='orange', width=0.4)
    plt.xlabel("Unique values in \'Proto\'")
    plt.ylabel("Percentage of value in column")
    plt.title("\'Proto\' Value Distribution")
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    state_val = state_df.index
    state_count = state_df.values
    plt.bar(state_val, state_count, color='blue', width=0.4)
    plt.xlabel("Unique values in \'State\'")
    plt.ylabel("Percentage of value in column")
    plt.title("\'State\' Value Distribution")

    plt.show()

    # plt.figure(figsize=(15, 10))
    # sns.pairplot(df, hue="Label")
    # plt.title("Looking for Insights in Data")
    # plt.legend("Label")
    # plt.tight_layout()
    # plt.plot()
    #
    # plt.figure(figsize=(15, 10))
    # sns.pairplot(df, hue="attack_cat")
    # plt.title("Looking for Insights in Data")
    # plt.legend("attack_cat")
    # plt.tight_layout()
    # plt.plot()



def random_forest_classifier_label(X_train, y_train, y_train_cat, X_val, y_val, y_val_cat,
                                   X_test, y_test_label, y_test_cat):

    clf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=random_seed,  n_jobs=-1)
    # clf = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=random_seed,
    #                                   min_samples_split=20, max_depth=140, max_features=None, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print("\nRF Label -------------------------------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_val, y_pred=y_pred) * 100))
    # display(HTML(pd.DataFrame(metrics.confusion_matrix(y_val, y_pred), columns=["Normal", "Attack"],
    #                           index=["Normal", "Attack"]).to_html()))
    print(metrics.classification_report(y_true=y_val, y_pred=y_pred, digits=4))

    # graph = Source(tree.export_graphviz(clf.estimators_[0], out_file=None, feature_names=X_train.columns))
    # graph.format = 'png'
    # graph.render('rand_tree_render', view=True)

    return

    feature_col = X_train.columns.to_list()
    random_forest_graph(clf, feature_col, "Random Forest Feature Importance (Label)")


def random_forest_classifier_atk_cat(X_train, y_train, y_train_cat, X_val, y_val, y_val_cat,
                                     X_test, y_test_label, y_test_cat):
    clf_cat = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=random_seed, n_jobs=-1)
    # clf_cat = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=random_seed,
    #                                   min_samples_split=20, max_depth=140, max_features=None, n_jobs=-1)
    clf_cat.fit(X_train, y_train_cat)
    y_pred_cat = clf_cat.predict(X_val)

    print("\nRF Atk Cat -------------------------------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_val_cat, y_pred=y_pred_cat) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
                      'Analysis', 'Shellcode', 'Worms']
    # display(HTML(pd.DataFrame(metrics.confusion_matrix(y_val_cat, y_pred_cat), columns=attack_columns,
    #                           index=attack_columns).to_html()))
    print(metrics.classification_report(y_true=y_val_cat, y_pred=y_pred_cat, labels=attack_columns, digits=4))
    # print(metrics.classification_report(y_true=y_val_cat, y_pred=y_pred_cat))

    # graph = Source(tree.export_graphviz(clf_cat.estimators_[0], out_file=None, feature_names=X_train.columns))
    # graph.format = 'png'
    # graph.render('rand_tree_cat_render', view=True)

    return

    feature_col = X_train.columns.to_list()
    random_forest_graph(clf_cat, feature_col, "Random Forest Feature Importance (Attack Category)")


def random_forest_classifier_RFECV(X_train, y_train, y_train_cat, X_val, y_val, y_val_cat):
    y_train = y_train.to_numpy().ravel()
    y_train_cat = y_train_cat.to_numpy().ravel()

    # clf_cat = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=random_seed)
    # clf_cat.fit(X_train, y_train_cat)
    # y_pred_cat = clf_cat.predict(X_val)
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    # model_label = RFECV(estimator=DecisionTreeClassifier(random_state=random_seed, criterion='entropy'),
    #                   scoring="accuracy",
    #                   min_features_to_select=1,
    #                   n_jobs=-1,
    #                   step=3)
    # print("fit")
    # model_lab = model_label.fit(X_train, y_train)
    # print("model_label")
    # print(model_lab)
    # X_train_rfe_lab = model_label.transform(X_train)
    # X_val_rfe_lab = model_label.transform(X_val)
    # clf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=random_seed)
    # clf.fit(X_train_rfe_lab, y_train)
    # y_pred = clf.predict(X_val_rfe_lab)

    # print("\nRF Label -------------------------------------------------")
    # print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_val, y_pred=y_pred) * 100))
    # print(pd.DataFrame(metrics.confusion_matrix(y_val, y_pred), columns=["Normal", "Attack"], index=["Normal", "Attack"]))
    # print(metrics.classification_report(y_true=y_val, y_pred=y_pred))

    model_cat = RFECV(estimator=DecisionTreeClassifier(random_state=random_seed, criterion='entropy'),
                      scoring="f1_macro",
                      min_features_to_select=1,
                      n_jobs=-1,
                      step=3)
    print("fit")
    model = model_cat.fit(X_train, y_train_cat)
    print("model")
    print(model)
    X_train_rfe = model_cat.transform(X_train)
    X_val_rfe = model_cat.transform(X_val)

    # clf_cat2 = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=random_seed)
    # clf_cat2 = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=random_seed,
    #                                   min_samples_split=20, max_depth=140, max_features=None)
    clf_cat2 = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=random_seed)
    clf_cat2.fit(X_train_rfe, y_train_cat)
    y_pred_cat = clf_cat2.predict(X_val_rfe)

    print("\nRF Atk Cat -------------------------------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_val_cat, y_pred=y_pred_cat) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
                      'Analysis', 'Shellcode', 'Worms']
    # print(metrics.confusion_matrix(y_val_cat, y_pred_cat))
    print(metrics.classification_report(y_true=y_val_cat, y_pred=y_pred_cat, labels=attack_columns, digits=4))

    n_scores = len(model_cat.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(1, n_scores + 1),
        model_cat.cv_results_["mean_test_score"],
        yerr=model_cat.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()


    # feature_col = X_train.columns.to_list()
    # random_forest_graph(clf, feature_col, "Random Forest Feature Importance (Label)")
    # random_forest_graph(clf_cat, feature_col, "Random Forest Feature Importance (Attack Category)")


def random_forest_OOB_error_rate(X_train, y_train, y_train_cat, X_val, y_val, y_val_cat,
                                     X_test, y_test_label, y_test_cat):
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
    plt.title("Forest OOB Error vs Forest Size (Atk Category)")
    plt.show()


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


def random_forest_randomizedSearchCV(X_train, y_train, y_train_cat, X_val, y_val, y_val_cat):

    y_train_1d = y_train.to_numpy().ravel()
    y_train_cat_1d = y_train_cat.to_numpy().ravel()

    clf = RandomForestClassifier(random_state=random_seed, criterion="entropy")

    max_depth = [int(x) for x in np.linspace(20, 220, num=11)]
    max_depth.append(None)
    random_grid = {'n_estimators':[int(x) for x in np.linspace(start=100, stop=300, num=3)],
                   'max_features': ['sqrt', 'log2', None],
                   'max_depth': max_depth,
                   'min_samples_split': [2, 5, 10, 20],
                   'min_samples_leaf': [1, 2, 4, 8],
                   'class_weight': ['balanced', 'balanced_subsample', None],
                   'bootstrap': [True, False]}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=10, cv=3, verbose=2,
                                   random_state=random_seed, n_jobs=-1)  # Fit the random search model
    rf_random.fit(X_train, y_train_cat_1d)

    print("Best Params ----------------------")
    print(rf_random.best_params_)

    clf_cat = rf_random.best_estimator_

    y_pred_cat = clf_cat.predict(X_val)

    print("\nBest RF Atk Cat -------------------------------------------------")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_true=y_val_cat, y_pred=y_pred_cat) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoor',
                      'Analysis', 'Shellcode', 'Worms']
    # display(HTML(pd.DataFrame(metrics.confusion_matrix(y_val_cat, y_pred_cat), columns=attack_columns,
    #                           index=attack_columns).to_html()))
    print(metrics.classification_report(y_true=y_val_cat, y_pred=y_pred_cat, labels=attack_columns))
    # print(metrics.classification_report(y_true=y_val_cat, y_pred=y_pred_cat))


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


def save_model(clf, X_val, y_val):
    pickle.dump(clf, open('rf_model.pickle', 'wb'))
    pickle.dump(X_val, open('rf_x_val.pickle', 'wb'))
    pickle.dump(y_val, open('rf_y_val.pickle', 'wb'))


if __name__ == '__main__':
    start_time = time.time()
    read_data("UNSW-NB15-BALANCED-TRAIN.csv")
    print("END")
    print(f"--- {(time.time() - start_time):.2f} seconds ---")

