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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class DataProcessing:
    aminoacids = ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T']
    data = None
    raw_sequences = None
    encoded_sequences = None
    unscaled_values = None
    scaled_values = None
    label_encoder: LabelEncoder = LabelEncoder().fit(np.array(aminoacids))
    onehot_encoder: OneHotEncoder = OneHotEncoder(sparse=False, dtype=int)
    scaler: StandardScaler = None
    split = None

    def __init__(self, filepath):
        data = pd.DataFrame(pd.read_csv(filepath), columns=['Sequence', 'Data']).to_numpy()
        self.data = data

        self.unscaled_values = np.array([entry[1] for entry in data]).reshape(-1, 1)
        self.scaler = StandardScaler().fit(self.unscaled_values)
        self.scaled_values = self.scaler.transform(self.unscaled_values)

        self.raw_sequences = np.array([entry[0] for entry in data])
        self.encoded_sequences = np.array([DataProcessing.encode(pseq) for pseq in self.raw_sequences])

        self.split = train_test_split(self.encoded_sequences, self.scaled_values)

    @staticmethod
    def encode(pseq):
        pseq = np.array(list(pseq.upper()))
        int_enc = DataProcessing.label_encoder.transform(pseq)
        int_enc = int_enc.reshape(len(int_enc), 1)
        onehot_enc = DataProcessing.onehot_encoder.fit_transform(int_enc)
        return onehot_enc

    @staticmethod
    def decode(encoded_pseq):
        decoded = DataProcessing.onehot_encoder.inverse_transform(encoded_pseq)
        decoded = DataProcessing.label_encoder.inverse_transform(np.ravel(decoded))
        sequence = ""
        for char in decoded:
            sequence += char
        return sequence

    def unscale_value(self, scaled_value):
        unscaled = self.scaler.inverse_transform([scaled_value]).flatten()
        return unscaled
