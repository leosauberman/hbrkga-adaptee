import copy
import random
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

import numpy
import numpy as np
from joblib import Parallel

from hbrkga.brkga_mp_ipr.algorithm import BrkgaMpIpr
from hbrkga.brkga_mp_ipr.enums import Sense
from hbrkga.brkga_mp_ipr.types import BaseChromosome
from hbrkga.brkga_mp_ipr.types_io import load_configuration
from hbrkga.exploitation_method_BO_only_elites import BayesianOptimizerElites
from sklearn import clone, svm, datasets
from sklearn.base import is_classifier
from sklearn.datasets import make_classification
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import check_cv, train_test_split
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid, GridSearchCV
from sklearn.model_selection._validation import _fit_and_score, _warn_or_raise_about_fit_failures, _insert_error_scores, \
    cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import indexable
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import _check_fit_params


class Decoder:

    def __init__(self, parameters, estimator, X, y, cv):
        self._parameters = parameters
        self._estimator = estimator
        self._X = X
        self._y = y
        self._limits = [self._parameters[l] for l in list(self._parameters.keys())]
        self._cv = cv

    def decode(self, chromosome: BaseChromosome, rewrite: bool) -> float:
        return self.score(self.encoder(chromosome))

    def encoder(self, chromosome: BaseChromosome) -> dict:
        chr_size = len(chromosome)
        hyperparameters = copy.deepcopy(self._parameters)

        for geneIdx in range(chr_size):
            gene = chromosome[geneIdx]
            key = list(self._parameters.keys())[geneIdx]
            limits = self._parameters[key]  # evita for's aninhados
            if type(limits) is np.ndarray:
                limits = limits.tolist()

            if type(limits[0]) is str:
                hyperparameters[key] = limits[round(gene * (len(limits) - 1))]
            elif type(limits[0]) is int and len(limits) > 2:
                hyperparameters[key] = int(limits[round(gene * (len(limits) - 1))])
            elif type(limits[0]) is bool:
                hyperparameters[key] = 1 if limits[0] else 0
            else:
                hyperparameters[key] = (gene * (limits[1] - limits[0])) + limits[0]

        return hyperparameters

    def score(self, hyperparameters: dict) -> float:
        estimator_clone = clone(self._estimator)
        estimator_clone.set_params(**hyperparameters)

        try:
            estimator_clone.fit(self._X, self._y)
        except ValueError:
            return 0.0

        # return estimator_clone.score(self._X, self._y)
        return cross_val_score(estimator_clone, self._X, self._y, cv=self._cv).mean()


class HyperBRKGASearchCV(BaseSearchCV):

    def __init__(
            self,
            estimator,
            *,
            scoring=None,
            n_jobs=None,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch="2*n_jobs",
            error_score=np.nan,
            return_train_score=True,
            parameters,
            data,
            target
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.brkga_config, _ = load_configuration("./hbrkga/config.conf")
        self._parameters = parameters

        self.decoder = Decoder(self._parameters, estimator, data, target, cv)
        elite_number = int(self.brkga_config.elite_percentage * self.brkga_config.population_size)
        self.em_bo = BayesianOptimizerElites(decoder=self.decoder, e=0.3, steps=3, percentage=0.6,
                                             eliteNumber=elite_number)
        chromosome_size = len(self._parameters)
        self.brkga = BrkgaMpIpr(
            decoder=self.decoder,
            sense=Sense.MAXIMIZE,
            seed=random.randint(-10000, 10000),
            chromosome_size=chromosome_size,
            params=self.brkga_config,
            diversity_control_on=True,
            n_close=3,
            exploitation_method=self.em_bo
        )

        self.brkga.initialize()

    def fit(self, X, y=None, *, groups=None, **fit_params):
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )

        def evaluate_candidates(candidate_params, cv=None, more_results=None):
            start = datetime.now()
            cv = cv or cv_orig
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)
            all_candidate_params = []
            all_more_results = defaultdict(list)

            for i in range(1, 11):
                print("\n###############################################")
                print(f"Generation {i}")
                print("")
                self.brkga.evolve()

                for pop_idx in range(len(self.brkga._current_populations)):
                    pop_diversity_score = self.brkga.calculate_population_diversity(pop_idx)
                    if self.verbose > 2:
                        print(f"Population {pop_idx}:")
                        print(f"Population diversity score = {pop_diversity_score}")
                        print("")
                        print("Chromosomes = ")
                        for chromo_idx in range(len(self.brkga._current_populations[pop_idx].chromosomes)):
                            print(f"{chromo_idx} -> {self.brkga._current_populations[pop_idx].chromosomes[chromo_idx]}")
                        print("")
                        print("Fitness = ")
                        for fitness in self.brkga._current_populations[pop_idx].fitness:
                            print(fitness)
                        print("------------------------------")

                best_cost = self.brkga.get_best_fitness()
                best_chr = self.brkga.get_best_chromosome()
                if self.verbose > 2:
                    print(f"{datetime.now()} - Best score so far: {best_cost}")
                    print(f"{datetime.now()} - Best chromosome so far: {best_chr}")
                    print(f"{datetime.now()} - Total time so far: {datetime.now() - start}", flush=True)

            best_cost = self.brkga.get_best_fitness()
            best_chr = self.brkga.get_best_chromosome()
            if self.verbose > 2:
                print("\n###############################################")
                print("Final results:")
                print(f"{datetime.now()} - Best score: {best_cost}")
                print(f"{datetime.now()} - Best chromosome: {best_chr}")
                print(f"Total time = {datetime.now() - start}")

            all_candidate_params.extend(candidate_params)
            self.results = {
                "best_chromosome": best_chr,
                "best_param_decoded": self.decoder.encoder(best_chr),
                "best_param_score": best_cost,
                "total_time": (datetime.now() - start).total_seconds(),
            }

        self._run_search(evaluate_candidates)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = self.results
        self.n_splits_ = n_splits

        return self

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(ParameterGrid(self._parameters))


"""
# Exemplo 1
if __name__ == '__main__':
    iris = datasets.load_iris()
    params = {'C': [1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}

    svc = svm.SVC()

    clf = HyperBRKGASearchCV(svc, parameters=params, data=iris.data, target=iris.target, verbose=3)

    clf.fit(iris.data, iris.target)
    print(clf.cv_results_)
"""

# Exemplo 2
if __name__ == '__main__':
    param_grid = {
        'max_depth': [2, 4, 8, 16, 32, 64],
        'min_samples_leaf': [2, 4, 8, 16],
        'criterion': ['gini', 'entropy', 'log_loss']
    }

    tree = DecisionTreeClassifier()
    # treeHBRKGA = DecisionTreeClassifier(criterion='entropy', max_depth=64, min_samples_leaf=2)
    # treeGS = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=4)
    iris = datasets.load_iris()

    # hyperbrkga = HyperBRKGASearchCV(tree, parameters=param_grid, cv=10, scoring='accuracy',
    #                                 data=iris.data, target=iris.target, refit=True, verbose=3)
    # hyperbrkga.fit(iris.data, iris.target)
    # print('HyperBRKGA')
    # print(hyperbrkga.cv_results_)
    # print('--------------------------------------------\n')
    #
    grid = GridSearchCV(tree, param_grid)
    grid.fit(iris.data, iris.target)
    print('GridSearch')
    print(grid.best_params_, grid.best_score_)
    mean_fit_time = grid.cv_results_['mean_fit_time']
    mean_score_time = grid.cv_results_['mean_score_time']
    n_splits = grid.n_splits_  # number of splits of training data
    import pandas as pd
    n_iter = pd.DataFrame(grid.cv_results_).shape[0]  # Iterations per split

    print(np.mean(mean_fit_time + mean_score_time) * n_splits * n_iter)

    # X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1, stratify=iris.target)
    #
    # treeHBRKGA.fit(X_train, y_train)
    # treeGS.fit(X_train, y_train)
    #
    # print(treeHBRKGA.score(X_test, y_test))
    # print(treeGS.score(X_test, y_test))



"""
HyperBRKGA
'best_param_decoded': 
{'criterion': 'entropy', 'max_depth': 64, 'min_samples_leaf': 2},
'best_param_score': 0.98, 
'total_time': datetime.timedelta(seconds=1, microseconds=725785)}

Grid Search
{'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 4}

"""


