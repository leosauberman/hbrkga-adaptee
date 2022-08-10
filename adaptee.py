import copy
import random
from pprint import pprint

import numpy as np

from hbrkga.brkga_mp_ipr.algorithm import BrkgaMpIpr
from hbrkga.brkga_mp_ipr.enums import Sense
from hbrkga.brkga_mp_ipr.types import BaseChromosome
from hbrkga.brkga_mp_ipr.types_io import load_configuration
from hbrkga.exploitation_method_BO_only_elites import BayesianOptimizerElites
from sklearn import clone, svm, datasets
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid


class SVCDecoder:

    def __init__(self, parameters, estimator, X, y):
        self._parameters = parameters
        self._estimator = estimator
        self._X = X
        self._y = y
        self._limits = [self._parameters[l] for l in list(self._parameters.keys())]

    def decode(self, chromosome: BaseChromosome, rewrite: bool) -> float:
        chr_size = len(chromosome)
        hyperparameters = copy.deepcopy(self._parameters)

        for geneIdx in range(chr_size):
            gene = chromosome[geneIdx]
            # if type(gene) is str:
            #     value = sum([ord(x) for x in gene])
            key = list(self._parameters.keys())[geneIdx]
            limits = self._parameters[key]  # evita for's aninhados
            hyperparameters[key] = (gene * (limits[1] - limits[0])) + limits[0]

        return self.score(hyperparameters)

    def score(self, hyperparameters: dict) -> float:
        estimator_clone = clone(self._estimator)
        estimator_clone.set_params(**hyperparameters)

        estimator_clone.fit(self._X, self._y)

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

        self.decoder = SVCDecoder(self._parameters, estimator, data, target)
        elite_number = int(self.brkga_config.elite_percentage * self.brkga_config.population_size)
        self.em_bo = BayesianOptimizerElites(decoder=self.decoder, e=0.3, steps=3, percentage=0.6, eliteNumber=elite_number)
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

    def _run_search(self, evaluate_candidates):
        # print(evaluate_candidates)
        evaluate_candidates(ParameterGrid(self._parameters))


if __name__ == '__main__':
    iris = datasets.load_iris()
    params = {'gamma': [1e-4, 1e-1], 'C': [1, 10]}
    svc = svm.SVC()

    clf = HyperBRKGASearchCV(svc, parameters=params, data=iris.data, target=iris.target)

    # clf = GridSearchCV(svc, parameters)
    clf.fit(iris.data, iris.target)
    print(clf.cv_results_)
