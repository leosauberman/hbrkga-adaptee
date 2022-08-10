from datetime import datetime

from hbrkga.brkga_mp_ipr.enums import Sense
from hbrkga.brkga_mp_ipr.types_io import load_configuration
from hbrkga.brkga_mp_ipr.algorithm import BrkgaMpIpr

from hbrkga.nn_instance_PT import NNInstance
from hbrkga.nn_decoder_PT import NNDecoder
import random

from exploitation_method_BO_only_elites import BayesianOptimizerElites

import sys

start = datetime.now()

instance = NNInstance(
        #train_path = sys.argv[1],
        #test_path = sys.argv[2],
        train_path = "./datasets/rectangles_train.csv",
        test_path = "./datasets/rectangles_val.csv",
        epochs_no = 300,
        batch_size = 50000)

brkga_params, _ = load_configuration("./config.conf")

eliteNumber = int(brkga_params.elite_percentage * brkga_params.population_size)

decoder = NNDecoder(
        instance = instance,
        limits = [(5,15), (5,30), (5,45), (0.000001,0.1), (0,0.001)])

EM_BO = BayesianOptimizerElites(decoder = decoder, e = 0.3, steps = 3, percentage = 0.6, eliteNumber = eliteNumber)

brkga = BrkgaMpIpr(
        decoder=decoder,
        sense=Sense.MAXIMIZE,
        seed=random.randint(-10000,10000),
        chromosome_size=instance._num_nodes,
        params=brkga_params,
        diversity_control_on = True,
        n_close = 3,
        exploitation_method= EM_BO)

brkga.initialize()

for i in range(1, 11):
        print("\n###############################################")
        print(f"Generation {i}")
        print("")
        brkga.evolve()

        for pop_idx in range(len(brkga._current_populations)):
                print(f"Population {pop_idx}:")
                print(f"Population diversity score = {brkga.calculate_population_diversity(pop_idx)}")
                print("")
                print("Chromosomes = ")
                for chromo_idx in range(len(brkga._current_populations[pop_idx].chromosomes)):
                        print(f"{chromo_idx} -> {brkga._current_populations[pop_idx].chromosomes[chromo_idx]}")
                print("")
                print("Fitness = ")
                for fitness in brkga._current_populations[pop_idx].fitness:
                        print(fitness)
                print("------------------------------")

        best_cost = brkga.get_best_fitness()
        print(f"{datetime.now()} - Best score so far: {best_cost}")
        best_chr = brkga.get_best_chromosome()
        print(f"{datetime.now()} - Best chromosome so far: {best_chr}")
        print(f"{datetime.now()} - Total time so far: {datetime.now() - start}", flush = True)

print("\n###############################################")
print("Final results:")
best_cost = brkga.get_best_fitness()
print(f"{datetime.now()} - Best score: {best_cost}")
best_chr = brkga.get_best_chromosome()
print(f"{datetime.now()} - Best chromosome: {best_chr}")
print(f"Total time = {datetime.now() - start}")