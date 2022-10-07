import copy
import random
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

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
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid
from sklearn.model_selection._validation import _fit_and_score, _warn_or_raise_about_fit_failures, _insert_error_scores
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import indexable
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import _check_fit_params


class Decoder:

    def __init__(self, parameters, estimator, X, y):
        self._parameters = parameters
        self._estimator = estimator
        self._X = X
        self._y = y
        self._limits = [self._parameters[l] for l in list(self._parameters.keys())]

    def decode(self, chromosome: BaseChromosome, rewrite: bool) -> float:
        return self.score(self.encoder(chromosome))

    def encoder(self, chromosome: BaseChromosome) -> dict:
        chr_size = len(chromosome)
        hyperparameters = copy.deepcopy(self._parameters)

        for geneIdx in range(chr_size):
            gene = chromosome[geneIdx]
            key = list(self._parameters.keys())[geneIdx]
            limits = self._parameters[key]  # evita for's aninhados
            if type(limits[0]) is str:
                hyperparameters[key] = limits[round(gene * (len(limits) - 1))]
            elif type(limits[0]) is int and len(limits) > 2:
                hyperparameters[key] = int(limits[round(gene * (len(limits) - 1))])
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

        return estimator_clone.score(self._X, self._y)


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

        self.decoder = Decoder(self._parameters, estimator, data, target)
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

    def fit_new(self, X, y=None, *, groups=None, **fit_params):
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
                "total_time": (datetime.now() - start),
            }

        self._run_search(evaluate_candidates)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = self.results
        self.n_splits_ = n_splits

        return self

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

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

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
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                    )
                )
                print('---------------------------------------------')
                print(out)
                print('---------------------------------------------\n')

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(ParameterGrid(self._parameters))


"""
# Exemplo 1
if __name__ == '__main__':
    iris = datasets.load_iris()
    params = {'C': [1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}
    # Como lidar com casos como esse:
    # degree: int, default=3
    # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

    svc = svm.SVC()

    clf = HyperBRKGASearchCV(svc, parameters=params, data=iris.data, target=iris.target, verbose=3)

    clf.fit_new(iris.data, iris.target)
    print(clf.cv_results_)
"""

"""
# Exemplo 2
if __name__ == '__main__':
    param_grid = {
        'max_depth': [2, 4, 8, 16, 32, 64],
        'min_samples_leaf': [2, 4, 8, 16]
    }

    tree = DecisionTreeClassifier()

    np.random.seed(1)
    X, y = make_classification(n_samples=400, n_features=6, n_informative=6,
                               n_redundant=0, n_classes=10, class_sep=2)

    hyperbrkga = HyperBRKGASearchCV(tree, parameters=param_grid, cv=10, scoring='accuracy',
                                    data=X, target=y, refit=True, verbose=3)
    hyperbrkga.fit_new(X, y)
    print(hyperbrkga.cv_results_)
"""
