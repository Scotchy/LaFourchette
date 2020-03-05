import numpy as np

N_SPECILIATIES = 151
min_lf_prob = 0.9
sigmoid = lambda x : 1 / (1 + np.exp(-x))

eating_hours = [11, 12, 13, 14, 18, 19, 20, 21, 22]
commissions = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30]