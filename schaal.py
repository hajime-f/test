"""テストプログラムです"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def func_1(input_x1, input_x2):
    """Schaal関数の実装"""

    output_y = []

    for x_1, x_2 in zip(input_x1, input_x2):

        tmp = np.sqrt(np.power(x_1, 2) + np.power(x_2, 2))
        y_i = np.sin(tmp) / tmp

        output_y.append(y_i)

    return np.array(output_y)


N = 2
T = 1000
data_x = 20 * np.random.rand(T, N) - 10
data_y = func_1(data_x[:, 0], data_x[:, 1])

xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, test_size=0.1)

wire_x1 = np.linspace(-10, 10, 100)
wire_x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(wire_x1, wire_x2)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

ax.plot_wireframe(X1, X2, func_1(X1, X2), alpha=0.5)
ax.scatter(xtrain[:, 0], xtrain[:, 1], ytrain, color='red', label='label')

plt.show()
