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

import numpy as np
import keras
from itertools import combinations as combinator
from itertools import product
from operator import itemgetter


def run(pseqs, model: keras.models.Sequential):
    point_mutation_library, point_mutation_positions = _build_single_point_mutation_library_(pseqs)
    new_mutant_library = _cross_mutate_(point_mutation_library, point_mutation_positions)

    print("Evaluating cross mutants.")
    scores = []
    for new_mutant in new_mutant_library:
        score = model.predict(new_mutant).flatten()
        entries = []
        for pair in zip(new_mutant, score):
            entries.append(pair)
        scores.append(entries)

    best_of_each = []
    for score in scores:
        best = max(score, key=itemgetter(1))
        best_of_each.append(best)

    new_best = max(best_of_each, key=itemgetter(1))
    print("Optimal mutant deduced.")
    return new_best


def _build_single_point_mutation_library_(pseqs):
    num_samples = pseqs.shape[0]
    pep_len = pseqs.shape[1]
    aminos = 20

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
        x = np.argmax(deltas[i])
        most_freq[0, i, x] = 1

    single_point_mutation_library = [most_freq]
    for i, aa in enumerate(aa_options):
        if len(aa) > 1:
            point_mutations = []
            to_mutate = np.copy(most_freq[0])
            for a in aa:
                rigid = np.copy(a)
                rigid[rigid == 1] = 2
                to_mutate[i] = a
                mutated = np.copy(to_mutate)
                point_mutations.append(mutated)
            point_mutations = np.array(point_mutations)
            single_point_mutation_library.append(point_mutations)

    point_mutation_positions = [0]
    for i, aa in enumerate(aa_options):
        if len(aa) > 1:
            point_mutation_positions.append(i)

    print("Single point mutation library assembled.")
    return single_point_mutation_library, point_mutation_positions


def _cross_mutate_(single_point_mutation_library, point_mutation_positions):
    most_freq = single_point_mutation_library[0]

    for i in range(1, len(single_point_mutation_library)):
        mutant_set = single_point_mutation_library[i]
        position = point_mutation_positions[i]
        for mutant in mutant_set:
            aa = mutant[position]
            aa[aa == 1] = 2
            a = 0

    mutant_list_indices = np.arange(1, len(single_point_mutation_library))

    combinations = list(combinator(mutant_list_indices, 2))

    to_mix = []
    for combination in combinations:
        mixes = []
        for tup in combination:
            mixes.append(single_point_mutation_library[tup])
        to_mix.append(mixes)

    mutants_as_list = []
    for variant in to_mix:
        one_variant = []
        for mutant_set in variant:
            one_set = []
            for position in mutant_set:
                one_set.append(position)
            one_variant.append(one_set)
        mutants_as_list.append(one_variant)

    to_mutate = []
    for mutants in mutants_as_list:
        to_mutate.extend(product(*mutants))

    print("Creating cross mutants.")
    new_mutant_list = [tuple(most_freq)]
    for mutant_set in to_mutate:
        mutant = np.copy(mutant_set[0])
        for i in range(1, len(mutant_set)):
            polypep = mutant_set[i]
            for j, pep in enumerate(polypep):
                if 2 not in mutant[j]:
                    mutant[j] = pep
        stack = mutant_set + (mutant,)
        new_mutant_list.append(stack)

    for new_mutant in new_mutant_list:
        for aa in new_mutant:
            aa[aa == 2] = 1

    new_mutant_array = []
    for new_mutant in new_mutant_list:
        a = np.array([np.array(x) for x in new_mutant])
        new_mutant_array.append(a)

    new_mutant_array = np.array([np.array(y) for y in [np.array(x) for x in new_mutant_list]], dtype=object)

    print("Cross mutants assembled.")
    return new_mutant_array
