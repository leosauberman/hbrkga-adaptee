import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.model_selection._search import BaseSearchCV
from sklearn.neighbors import KNeighborsClassifier

sns.set()
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from adaptee import HyperBRKGASearchCV

if __name__ == "__main__":
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

    X = pd.DataFrame(diabetes_data_copy.drop(["Outcome"], axis=1),
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                              'BMI', 'DiabetesPedigreeFunction', 'Age'])
    y = diabetes_data_copy.Outcome
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    sc_X = StandardScaler()
    X_transformed = sc_X.fit_transform(X_train, )

    param_grid = {'n_neighbors': np.arange(1, 50)}

    knn = KNeighborsClassifier()

    file = open('results_2.csv', 'w', encoding="utf-8")


    def getCVTime(method: BaseSearchCV):
        mean_fit_time = method.cv_results_['mean_fit_time']
        mean_score_time = method.cv_results_['mean_score_time']
        n_splits = method.n_splits_  # number of splits of training data
        n_iter = pd.DataFrame(method.cv_results_).shape[0]  # Iterations per split
        return np.mean(mean_fit_time + mean_score_time) * n_splits * n_iter


    file.write("HyperBRKGAScore, HyperBRKGATime, HyperBRKGAParam, GridSearchScore, GridSearchTime, GridSearchParam, "
               "RandomSearchScore, RandomSearchTime, RandomSearchParam\n")

    hbrkga_mean = 0
    hbrkga_time_mean = 0
    grid_mean = 0
    grid_time_mean = 0
    random_mean = 0
    random_time_mean = 0

    for i in range(20):
        knn_hyper_cv = HyperBRKGASearchCV(knn, parameters=param_grid, cv=5, data=X_transformed, target=y_train)
        knn_hyper_cv.fit(X_transformed, y_train)
        hbrkga_score = knn_hyper_cv.cv_results_['best_param_score']
        hbrkga_time = knn_hyper_cv.cv_results_['total_time']
        hbrkga_param = knn_hyper_cv.cv_results_['best_param_decoded']['n_neighbors']
        hbrkga_mean += hbrkga_score
        hbrkga_time_mean += hbrkga_time

        knn_grid_cv = GridSearchCV(knn, param_grid, cv=5)
        knn_grid_cv.fit(X_transformed, y_train)
        grid_score = knn_grid_cv.best_score_
        grid_time = getCVTime(knn_grid_cv)
        grid_param = knn_grid_cv.best_params_['n_neighbors']
        grid_mean += grid_score
        grid_time_mean += grid_time

        knn_random_cv = RandomizedSearchCV(knn, param_grid, cv=5)
        knn_random_cv.fit(X_transformed, y_train)
        random_score = knn_random_cv.best_score_
        random_time = getCVTime(knn_random_cv)
        random_param = knn_random_cv.best_params_['n_neighbors']
        random_mean += random_score
        random_time_mean += random_time

        row = ", ".join([str(hbrkga_score), str(hbrkga_time), str(hbrkga_param), str(grid_score),
                         str(grid_time), str(grid_param), str(random_score), str(random_time), str(grid_param), "\n"])

        file.write(row)

    file.write(f"Média: {(hbrkga_mean / 20)}, Média: {(hbrkga_time_mean / 20)},, Média: {(grid_mean / 20)}, "
               f"Média: {(grid_time_mean / 20)},, Média: {(random_mean / 20)}, Média: {(random_time_mean / 20)},")

    file.close()

    new_knn = KNeighborsClassifier(hbrkga_param)
    new_knn.fit(X_transformed, y_train)
    print(new_knn.predict(X_test))
