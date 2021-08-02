import numpy as np
import random

M = 20
N = 5

mu = [2 * np.random.rand(N, 1) - 1 for _ in range(M)]
print(mu)


d = {'mu_' + str(i): mu[i] for i in range(M)}

file_name = './mu_params'
np.savez(file_name, **d)

d = np.load(file_name + '.npz')
mu = [d['mu_' + str(i)] for i in range(M)]

print(mu)
