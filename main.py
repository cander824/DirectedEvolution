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

import DataPreprocessing
import LayerOptimizer
import MutationGenerator
import os
import keras
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
file = 'proteindata.csv'


def main():
    pseqs, scaled_values = DataPreprocessing.run(file)
    opt_model = None
    split = train_test_split(pseqs, scaled_values)
    try:
        opt_model = keras.models.load_model("opt_model")
    except:
        opt_model = LayerOptimizer.run(split, scaled_values)

    train_x, test_x, train_y, test_y = split
    opt_model.evaluate(test_x, test_y)
    single_point_mutation_libary = MutationGenerator._build_single_point_mutation_library_(pseqs)
    optimals = MutationGenerator._find_optimal_point_mutations_(single_point_mutation_libary, opt_model)
    MutationGenerator._stack_mutations_(optimals, opt_model)
    print("Done")


if __name__ == "__main__":
    main()


