import sys

import numpy as np
# import TinyBrain as tb

male_height = np.random.normal(170, 5, 300)
female_height = np.random.normal(160, 5, 300)

male_weight = np.random.normal(120, 5, 300)
female_weight = np.random.normal(100, 5, 300)

male_label = [1] * 300
female_label = [0] * 300

train_set = np.array([np.concatenate((male_height, female_height)),
                    np.concatenate((male_weight, female_weight)),
                    np.concatenate((male_label, female_label))])

np.random.shuffle(train_set)

