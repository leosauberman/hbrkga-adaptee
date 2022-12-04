import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.model_selection._search import BaseSearchCV
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.preprocessing import StandardScaler
from adaptee import HyperBRKGASearchCV

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    diabetes_data = pd.read_csv('./hbrkga/datasets/diabetes.csv')
    diabetes_data_copy = diabetes_data.copy(deep=True)
    diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

    diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
    diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
    diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
    diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
    diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)

    # train_test_split
    # data leakage

    X = pd.DataFrame(diabetes_data_copy.drop(['Outcome'], axis=1),
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                              'BMI', 'DiabetesPedigreeFunction', 'Age'])
    y = diabetes_data_copy.Outcome
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    sc_X = StandardScaler()
    X_transformed = sc_X.fit_transform(X_train, )

    param_grid = {
        'criterion': ('gini', 'entropy', 'log_loss'),
        'splitter': ('best', 'random'),
        'max_depth': np.arange(0, 64),
        'min_samples_split': np.arange(0, 20),
        'min_samples_leaf': np.arange(0, 20)
    }

    tree = DecisionTreeClassifier()

    df = pd.DataFrame()


    def get_cv_time(method: BaseSearchCV):
        mean_fit_time = method.cv_results_['mean_fit_time']
        mean_score_time = method.cv_results_['mean_score_time']
        n_splits = method.n_splits_  # number of splits of training data
        n_iter = pd.DataFrame(method.cv_results_).shape[0]  # Iterations per split
        return np.mean(mean_fit_time + mean_score_time) * n_splits * n_iter


    for i in range(10):
        tree_hyper_cv = HyperBRKGASearchCV(tree, parameters=param_grid, cv=5, data=X_transformed, target=y_train)
        tree_hyper_cv.fit(X_transformed, y_train)
        hbrkga_score = tree_hyper_cv.cv_results_['best_param_score']
        hbrkga_time = tree_hyper_cv.cv_results_['total_time']
        hbrkga_param = str(tree_hyper_cv.cv_results_['best_param_decoded'])

        tree_grid_cv = GridSearchCV(tree, param_grid, cv=5)
        tree_grid_cv.fit(X_transformed, y_train)
        grid_score = tree_grid_cv.best_score_
        grid_time = get_cv_time(tree_grid_cv)
        grid_param = str(tree_grid_cv.best_params_)

        tree_random_cv = RandomizedSearchCV(tree, param_grid, cv=5)
        tree_random_cv.fit(X_transformed, y_train)
        random_score = tree_random_cv.best_score_
        random_time = get_cv_time(tree_random_cv)
        random_param = str(tree_random_cv.best_params_)

        row = {
            'HyperBRKGAScore': hbrkga_score,
            'HyperBRKGATime': hbrkga_time,
            'HyperBRKGAParam': hbrkga_param,
            'GridSearchScore': grid_score,
            'GridSearchTime': grid_time,
            'GridSearchParam': grid_param,
            'RandomSearchScore': random_score,
            'RandomSearchTime': random_time,
            'RandomSearchParam': random_param
        }
        df = df.append(row, ignore_index=True)

    means = {
        'HyperBRKGAScore': '',
        'HyperBRKGATime': '',
        'HyperBRKGAParam': '',
        'GridSearchScore': '',
        'GridSearchTime': '',
        'GridSearchParam': '',
        'RandomSearchScore': '',
        'RandomSearchTime': '',
        'RandomSearchParam': ''
    }
    for i in df.columns:
        if 'Param' not in i:
            means[i] = f'Média: {df[i].mean()}'

    stds = {
        'HyperBRKGAScore': '',
        'GridSearchScore': '',
        'RandomSearchScore': '',
    }
    for i in df.columns:
        if 'Score' in i:
            stds[i] = f'Desvio Padrão: {df[i].std()}'

    df = df.append(means, ignore_index=True)
    df = df.append(stds, ignore_index=True)

    df.to_csv('pima_indians/res_final.csv')

    # new_tree = DecisionTreeClassifier(hbrkga_param)
    # new_tree.fit(X_transformed, y_train)
    # print(new_tree.predict(X_test))
