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

import keras
import numpy as np
from sklearn.model_selection import train_test_split

num_epochs = 10
imax = 10
imin = 0
jmax = 10
jmin = 0


def run(pseqs, scaled_vals):
    input_shape = (pseqs.shape[1], pseqs.shape[2])
    batch_size = int(0.01 * pseqs.shape[0])

    split = train_test_split(pseqs, scaled_vals)
    train_x, test_x, train_y, test_y = split

    results = np.zeros((imax, jmax, 1))

    print("Optimizing number of layers")
    for i in range(imin, imax):
        for j in range(jmin, jmax):
            try:
                model = build_sequential_model(input_shape, i, j)
                model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, verbose=0)
                test_loss = model.evaluate(test_x, test_y, verbose=0)
                results[i, j, 0] = test_loss
            except ValueError:
                results[i, j, 0] = -1

    iopt = 0
    jopt = 0
    min_acc = np.amax(np.amax(results, axis=0))
    print("Evaluating results")
    for i in range(imin, imax):
        for j in range(jmin, jmax):
            acc = results[i, j, 0]
            if acc != -1:
                if acc < min_acc:
                    min_acc = acc
                    iopt = i
                    jopt = j

    print("i_opt = ", iopt, "; j_opt = ",  jopt)
    output = build_sequential_model(input_shape, iopt, jopt)
    output.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, verbose=0)
    return output, split


def build_sequential_model(input_shape, i, j):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    for k in range(i, j):
        u = 2 ** k
        model.add(keras.layers.Conv1D(u, 5, activation='sigmoid', use_bias=True))
        model.add(keras.layers.MaxPool1D(2))
        model.add(keras.layers.Dense(u, activation='sigmoid', use_bias=True))
        model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid', use_bias=True))

    opt = keras.optimizers.SGD(learning_rate=0.01)
    loss_func = keras.losses.MeanSquaredError()

    model.compile(optimizer=opt, loss=loss_func)

    return model
