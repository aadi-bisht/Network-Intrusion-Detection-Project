import numpy as np
import pandas as pd
import operator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


def main():
    df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False).fillna(0)
    df = df.drop('ct_ftp_cmd', axis=1)

    df['sport'] = df['sport'].apply(check_port_int).fillna(0)
    df['dsport'] = df['dsport'].apply(check_port_int).fillna(0)

    # Convert categorical features to numerical values
    categorical_features = ['proto', 'service', 'state', 'srcip', 'dstip']
    le = LabelEncoder()
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])

    labels, uniques = pd.factorize(df['attack_cat'])

    # Scale the numerical features
    numerical_features = ["dur", "Spkts", "Dpkts", "sbytes", "dbytes", "sttl", "dttl"]
    df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])

    # Perform correlation-based feature selection
    X = df.drop(columns=["Label", "attack_cat"])
    y = df["Label"]
    selector = SelectKBest(f_classif, k=46)
    selector.fit(X, y)
    mask = selector.get_support()
    selected_features = X.columns[mask]
    print("Selected Features for Label Classification: ", selected_features)
    f_scores = selector.scores_
    z = list(zip(X.columns, f_scores))
    res = sorted(z, key=operator.itemgetter(1), reverse=True)
    print(f"{res[0]}: {res[1]}")
    # # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[selected_features], y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(df[selected_features], y, test_size=0.5, random_state=42)
    #
    # # Define the parameter grid for GridSearchCV
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
        "activation": ["logistic", "relu"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [500, 1000, 2000]
    }
    #
    # # Create the MLP classifier object
    # mlp = MLPClassifier(random_state=42)
    mlp = MLPClassifier(random_state=42, solver='adam', max_iter=2000, hidden_layer_sizes=(100,), alpha=0.0001, activation='relu')
    # # Create the GridSearchCV object
    # grid_search = RandomizedSearchCV(estimator=mlp, param_distributions=param_grid, n_iter=20, cv=3, verbose=2,
    #                                  random_state=42, n_jobs=-1)
    #
    # # Fit the GridSearchCV object to the training data
    # grid_search.fit(X_train, y_train)
    #
    # # Print the best hyperparameters
    # print("Best Hyperparameters:", grid_search.best_params_)
    #
    # Predict on the test set using the best estimator
    # y_pred = grid_search.best_estimator_.predict(X_test)
    mlp.fit(X_train, y_train)
    # y_pred = mlp.predict(X_test)
    # # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=5, zero_division=1))
    # accuracy_score(y_test, y_pred, zero_division=1)
    attack_cat(df, uniques)



def attack_cat(df):
    X = df.drop(columns=["Label", "attack_cat"])
    y, uniques = pd.factorize(df["attack_cat"])
    selector = SelectKBest(f_classif, k=9)
    selector.fit(X, y)
    mask = selector.get_support()
    selected_features = X.columns[mask]
    print("Selected Features for Attack Classification: \n", selected_features)
    f_scores = selector.scores_
    z = list(zip(X.columns, f_scores))
    res = sorted(z, key=operator.itemgetter(1), reverse=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
    # mlp = MLPClassifier(random_state=42)
    # param_grid = {
    #         "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
    #         "activation": ["logistic", "relu"],
    #         "solver": ["adam", "sgd"],
    #         "alpha": [0.0001, 0.001, 0.01],
    #         "max_iter": [500, 1000, 2000]
    #     }
    # grid_search = RandomizedSearchCV(estimator=mlp, param_distributions=param_grid, n_iter=20, cv=3, verbose=2,
    #                                                                    random_state=42, n_jobs=-1)
    mlp = MLPClassifier(random_state=42, solver='adam', max_iter=2000, hidden_layer_sizes=(100,), alpha=0.0001, activation='relu')
    mlp.fit(X_train, y_train)
    # grid_search.fit(X_train, y_train)
    # y_pred = grid_search.best_estimator_.predict(X_test)

    # print("Best Hyperparameters:", grid_search.best_params_)
    y_pred = mlp.predict(X_val)
    print(classification_report(y_val, y_pred, zero_division=1, target_names=uniques))


# convert hexadecimal port values to integer
def check_port_int(value):
    try:
        return int(str(value), 0)
    except ValueError:
        return np.NaN


if __name__ == '__main__':
    main()
