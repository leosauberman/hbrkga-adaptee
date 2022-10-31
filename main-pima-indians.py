import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

    sc_X = StandardScaler()
    X = pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"], axis=1), ),
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                              'BMI', 'DiabetesPedigreeFunction', 'Age'])
    y = diabetes_data_copy.Outcome

    param_grid = {'n_neighbors': np.arange(1, 50)}

    knn = KNeighborsClassifier()

    file = open('results.csv', 'w')

    def getCVTime(method: BaseSearchCV):
        mean_fit_time = method.cv_results_['mean_fit_time']
        mean_score_time = method.cv_results_['mean_score_time']
        n_splits = method.n_splits_  # number of splits of training data
        n_iter = pd.DataFrame(method.cv_results_).shape[0]  # Iterations per split
        return np.mean(mean_fit_time + mean_score_time) * n_splits * n_iter

    for i in range(20):

        knn_hyper_cv = HyperBRKGASearchCV(knn, parameters=param_grid, cv=5, data=X, target=y)
        knn_hyper_cv.fit(X, y)
        hbrkga_score = knn_hyper_cv.cv_results_['best_param_score']
        hbrkga_time = knn_hyper_cv.cv_results_['total_time']

        knn_grid_cv = GridSearchCV(knn, param_grid, cv=5)
        knn_grid_cv.fit(X, y)
        grid_score = knn_grid_cv.best_score_
        grid_time = getCVTime(knn_grid_cv)

        knn_random_cv = RandomizedSearchCV(knn, param_grid, cv=5)
        knn_random_cv.fit(X, y)
        random_score = knn_random_cv.best_score_
        random_time = getCVTime(knn_random_cv)

        row = ", ".join([str(hbrkga_score), str(hbrkga_time), str(grid_score), str(grid_time), str(random_score), str(random_time)])
        file.write(row)

    file.close()
