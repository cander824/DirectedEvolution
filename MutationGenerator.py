#     Copyright 2020 Connnor Anderson
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from DataPreprocessing import aminoacids, onehot_encoder
import numpy as np
import keras
from itertools import combinations_with_replacement as combinator


def _build_single_point_mutation_library_(pseqs):
    num_samples = pseqs.shape[0]
    pep_len = pseqs.shape[1]
    aminos = len(aminoacids)

    shape = (pep_len, aminos)
    deltas = np.zeros(shape)

    for pseq in pseqs:
        for j, aa in enumerate(pseq):
            deltas[j] = np.add(deltas[j], aa)

    aa_options = []
    for delta in deltas:
        pot_aa = []
        for j, val in enumerate(delta):
            option = np.zeros(aminos)
            if val != 0:
                option[j] = 1
                pot_aa.append(option)
        aa_options.append(pot_aa)

    most_freq = np.zeros((1, pep_len, aminos))
    for i, aa in enumerate(aa_options):
        most_freq[0, i, np.argmax(aa)] = 1

    single_point_mutation_library = [most_freq]
    for i, aa in enumerate(aa_options):
        if len(aa) > 1:
            point_mutations = []
            to_mutate = np.copy(most_freq[0])
            for a in aa:
                to_mutate[i] = a
                mutated = np.copy(to_mutate)
                point_mutations.append(mutated)
            point_mutations = np.array(point_mutations)
            single_point_mutation_library.append(point_mutations)

    return single_point_mutation_library


def _find_optimal_point_mutations_(point_mutation_library, model: keras.models.Sequential):
    all_results = []
    for point_mutants in point_mutation_library:
        results = model.predict(point_mutants)
        all_results.append(zip(point_mutation_library, results[0]))

    optimals = [point_mutation_library[0]]
    for results in all_results:
        optimal = max(results, key=lambda i: i[1])[0]
        optimals.append(optimal)

    return optimals


def _stack_mutations_(optimal_point_mutations, model: keras.models.Sequential):
    combinations = []
    for i in range(1, len(optimal_point_mutations)+1):
        mixed = list(combinator(optimal_point_mutations, i))
        combinations.append(mixed)
    return None
