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

import LayerOptimizer
import MutationGenerator
import os
import keras
import numpy as np
from DataProcessing import DataProcessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
filepath = 'proteindata.csv'


def main():
    processor = DataProcessing(filepath)
    pseqs = processor.encoded_sequences
    split = processor.split
    scaled_vals = processor.scaled_values
    try:
        model = keras.models.load_model("model")
    except OSError:
        model = LayerOptimizer.run(split, scaled_vals)
    encoded_variant, scaled_score = MutationGenerator.run(pseqs, model)
    decoded_variant = processor.decode(encoded_variant)
    unscaled_score = processor.unscale_value(scaled_score)

    best_from_training_set = "MAPTLSEQTRQLVRASVPALQKHSVAISATMGRLLFERYPETRSLSELPERQLHKSASALLAYARSIDNPSALQAAIRRMVLSHARAGVQAVHYPLGWECLRDAIKEVLGPDATETLLQAWKEAYDFLAHLLSTKEAQVYAVLAE"
    encoded_best = np.array([processor.encode(best_from_training_set)])
    scaled_best = model.predict(encoded_best)
    unscaled_best = processor.unscale_value(scaled_best)

    print("Directed evolution variant score: ", unscaled_score)
    print("Best variant score: ", unscaled_best)


if __name__ == "__main__":
    main()
