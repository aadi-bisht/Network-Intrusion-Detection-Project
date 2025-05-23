import ipaddress
import warnings
import pickle
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE, RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics # is used to create classification results
from sklearn.tree import export_graphviz # is used for plotting the decision tree
from six import StringIO # is used for plotting the decision tree
from IPython.display import Image # is used for plotting the decision tree
from IPython.core.display_functions import display
from IPython.core.display import HTML # is used for showing the confusion matrix
import pydotplus # is used for plotting the decision tree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Maybe move read data to a separate file for all analysis technique files to call
def read_data(csv_file):
    # col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    unsw_data = pd.read_csv(csv_file)
    # print(unsw_data.dtypes)

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
    # print(unsw_data['ct_ftp_cmd'].head())
    # unsw_data['is_ftp_login'] = unsw_data['is_ftp_login'].astype('bool')
    # unsw_data['attack_cat'] = unsw_data['attack_cat'].fillna("normal_traffic")

    # Factorize categorical data
    # unsw_data[['proto', 'state', 'service', 'attack_cat']] = unsw_data[['proto', 'state', 'service', 'attack_cat']]\
    #     .apply(lambda x: pd.factorize(x)[0])
    unsw_data[['proto', 'state', 'service']] = unsw_data[['proto', 'state', 'service']] \
        .apply(lambda x: pd.factorize(x)[0])

    unsw_data['attack_cat'] = unsw_data['attack_cat'].astype('string').fillna('None')
    unsw_data['attack_cat'] = unsw_data['attack_cat'].apply(check_attack_cat) #.factorize()[0]
    # code, unique = unsw_data['attack_cat'].apply(check_attack_cat).factorize()
    # print(code)
    # print(unique)
    #
    # code, unique = unsw_data['proto'].factorize()
    # print(code)
    # print(unique)

    # unsw_data.to_csv('fillna_UNSW-NB15-BALANCED-TRAIN.csv', index=False)
    # print("saved")

    # Check all missing values in dataset are handled
    # print(unsw_data.isna().sum())
    # Verify all datatypes converted to numerical
    # print(unsw_data.dtypes)

    # print(unsw_data)
    # print(unsw_data.shape)

    feature_list = list(unsw_data)
    label = unsw_data['Label']
    attack_cat = unsw_data['attack_cat']
    feature_cols = unsw_data.drop(columns=['Label', 'attack_cat'])
    """ feature_cols = feature_cols[['ct_dst_src_ltm', 'tcprtt', 'Dpkts', 'sbytes', 'Sintpkt', 'is_ftp_login', 
        'Djit', 'Stime', 'Sjit', 'Stime', 'smeansz', 'dsport', 'dur', 'is_sm_ips_ports', 'sport',
        'Sload', 'proto', 'res_bdy_len', 'ct_flw_http_mthd', 'Djit', 'Dload', 'dstip']] """
    # print(label.head())
    # print(attack_cat.head())
    # print(feature_cols.head())
    RFE_model_eval(feature_cols, attack_cat)
    # K_nearest_neighbors_label(feature_cols, label)
    # K_nearest_neighbors_attack_cat(feature_cols, attack_cat)
    # KNN_load_and_predict()
    #RFE_feature_selection(feature_cols, attack_cat)
    #decision_tree_classifier_label(feature_cols, label)
    #decision_tree_classifier_attack_cat(feature_cols, attack_cat)


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
        if x == "Backdoor":
            x = "Backdoors"
        return x
    except ValueError:
        return np.NaN

def RFE_model_eval(feature_cols, y_label):
    X_train, X_temp, y_train, y_temp = train_test_split(feature_cols, y_label, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # get a list of models to evaluate
    def get_models():
        models = dict()
        rfe = RFE(estimator=LogisticRegression(max_iter=50), n_features_to_select=5)
        model = DecisionTreeClassifier()
        models['Logistic Regression'] = Pipeline(steps=[('s',rfe),('m',model)])
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

def Recursive_Feature_Elim(feature_cols, y_label):
    # Perform RFE, selecting 5 features
    rfe = RFE(estimator=DecisionTreeClassifier(), step=2,n_jobs=-1, n_features_to_select=5)
    rfe.fit(X_train, y_train)
    print(X_train.columns[(rfe.get_support())])
    X_train_selected = rfe.transform(X_train)
    X_val_selected = rfe.transform(X_val)    

def Recursive_Feature_Elim_CV(feature_cols, y_label):
    # Perform RFE with cross validation for automatic selection of the best features
    rfe = RFECV(estimator=DecisionTreeClassifier(), step=2,n_jobs=-1)
    rfe.fit(X_train, y_train)
    print(X_train.columns[(rfe.get_support())])
    X_train_selected = rfe.transform(X_train)
    X_val_selected = rfe.transform(X_val)
    return X_train_selected, X_val_selected

def K_nearest_neighbors_label(feature_cols, y_label):
    X_train, X_temp, y_train, y_temp = train_test_split(feature_cols, y_label, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5) 

    # Scale & normalize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # Grid search for hyper parameter tuning
    """ grid_params = {'n_neighbors': [29], 'weights': ['uniform', 'distance'], 'metric': ['minkowski','euclidean','manhattan']}
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=3, cv=3, n_jobs=-1, scoring='f1_macro')
    g_res = gs.fit(X_train, y_train)
    print(g_res.best_score_)
    print(g_res.best_params_) 
    print(g_res.cv_results_) """

    clf = KNeighborsClassifier(n_neighbors=6, metric='manhattan', p=1, weights='distance')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_val, y_pred) * 100))
    display(HTML(pd.DataFrame(metrics.confusion_matrix(y_val, y_pred), columns=["Normal", "Attack"],
                              index=["Normal", "Attack"]).to_html()))
    print(metrics.classification_report(y_val, y_pred))

def K_nearest_neighbors_attack_cat(feature_cols, y_label):
    X_train, X_temp, y_train, y_temp = train_test_split(feature_cols, y_label, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Perform feature selection using RFE
    """ min_features_to_select = 1
    rfe = RFECV(estimator=DecisionTreeClassifier(),step=2, min_features_to_select=min_features_to_select)
    rfe.fit(X_train, y_train)
    X_train = rfe.transform(X_train)
    X_val= rfe.transform(X_val)
        
    n_scores = len(rfe.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfe.cv_results_["mean_test_score"],
        yerr=rfe.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()

    rfe_df = pd.DataFrame({'Features': feature_cols.columns, 'Kept': rfe.support_, 'Rank': rfe.ranking_})
    rfe_df.to_csv('rfe_rank.csv')
 """
    # scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    
    #print(feature_cols.columns[rfe.support_])
    print(X_train.shape)
    print(X_val.shape)

    # finding optimal k using macro F1 scores
    """ acc = []
    for i in range (1,11):
        print("Training K value:",i)
        clf = KNeighborsClassifier(n_neighbors=i, metric='manhattan', p=1, weights='distance', n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc.append(metrics.f1_score(y_val,y_pred, average='macro')) """

    # plot macro-F1 score vs. K value
    """ plt.figure(figsize=(10,6))
    plt.plot(range(1,11),acc,color = 'blue', linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
    plt.title('F1 Score vs K value')
    plt.xlabel('K')
    plt.ylabel('F1 Score')
    plt.show()
    print("Maximum Accuracy:-",max(acc), "at K =",acc.index(max(acc)) + 1)  """

    # Grid search for hyper parameter tuning
    """ grid_params = {'n_neighbors': [4,5,6,7], 'weights': ['uniform', 'distance'], 'metric': ['minkowski','euclidean','manhattan']}
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=3, cv=3, n_jobs=-1, scoring='f1_macro')
    g_res = gs.fit(X_train, y_train)
    print(g_res.best_score_)
    print(g_res.best_params_) 
    print(g_res.cv_results_)
    df = pd.DataFrame(g_res.cv_results_)
    df.to_csv('knn_gridsearch_results.csv', index=False) """

    # train & predict using KNN
    n_neighbors = 6
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='manhattan', p=1, weights='distance')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_val, y_pred) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoors',
                      'Analysis', 'Shellcode', 'Worms']
    print("Classifier: K-Nearest Neighbors")
    print(metrics.classification_report(y_val, y_pred, labels=attack_columns))
    
    # pickle model/dataset
    """ pickle.dump(clf, open('knn_model.pickle', 'wb'))
    pickle.dump(X_val, open('x_val.pickle','wb'))
    pickle.dump(y_val, open('y_val.pickle','wb')) """


    

def KNN_load_and_predict():
    # cmap_light = ListedColormap(['red', 'blue', 'lime', 'salmon', 'orange', 'lightyellow', 'fuschia', 'cyan', 'violet', 'skyblue'])
    # cmap_dark = ListedColormap(['darkred','darkblue','darkgreen','darksalmon','darkorange','gold','purple','teal','darkviolet', 'dodgerblue'])

    # load model/dataset
    clf2 = pickle.load(open('knn_model.pickle', 'rb'))
    X_val = pickle.load(open('x_val.pickle', 'rb'))
    y_val = pickle.load(open('y_val.pickle', 'rb'))


    y_pred = clf2.predict(X_val)

    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_val, y_pred) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoors',
                      'Analysis', 'Shellcode', 'Worms']
    display(HTML(pd.DataFrame(metrics.confusion_matrix(y_val, y_pred), columns=attack_columns,
                              index=attack_columns).to_html()))
    print(metrics.classification_report(y_val, y_pred, labels=attack_columns))

def decision_tree_classifier_label(feature_cols, y_label):
    X_train, X_temp, y_train, y_temp = train_test_split(feature_cols, y_label, test_size=0.1, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # print(y_train.shape)
    # print(y_val.shape)
    # print(y_test.shape)

    clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    display(HTML(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred), columns=["Normal", "Attack"],
                              index=["Normal", "Attack"]).to_html()))
    print(metrics.classification_report(y_test, y_pred))


def decision_tree_classifier_attack_cat(feature_cols, y_label):
    X_train, X_temp, y_train, y_temp = train_test_split(feature_cols, y_label, test_size=0.1, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
    attack_columns = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoors',
                      'Analysis', 'Shellcode', 'Worms']
    display(HTML(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred), columns=attack_columns,
                              index=attack_columns).to_html()))
    print(metrics.classification_report(y_test, y_pred, labels=attack_columns))


if __name__ == '__main__':
    read_data("UNSW-NB15-BALANCED-TRAIN.csv")
