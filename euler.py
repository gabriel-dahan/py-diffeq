# %% TESTS - IGNORE | Not part of the module %% #

import matplotlib.pyplot as plt
from math import exp

# %% Résolution d'équations différentielles par la méthode d'Euler. %%

# % 1er ordre %

def euler(N, a, b, f, x0) -> tuple[list]:
    t, x = [a], [x0]
    h = (b - a) / (N - 1)
    
    for i in range(N - 1):
        t.append(a + h * (i + 1))
        x.append(x[i] + h * f(x[i]))

    return t, x

def test_euler1():
    TAU = 0.05
    E = 1

    x, y = euler(100, 0, 6 * TAU, lambda u: (E - u) / TAU, 0)
    plt.plot(x, y, '+', label = 'Méthode d\'Euler')
    plt.plot(x, [E * (1 - exp(-x[i] / TAU)) for i in range(len(x))], label = 'Solution exacte')
    plt.show()

# % 2ème ordre (ou plus) %

from scipy.integrate import odeint
import numpy as np

def f(X, t):
    return [X[1], -X[0]]

def f2(X, t):
    return [X[1], 0.2 * X[1] - X[0]]

def graphic_sol(f, init, range_):

    t = np.linspace(range_[0], range_[1], 1000)
    sol = odeint(f, init, t)
    x = [el[0] for el in sol]
    plt.plot(t, x)
    plt.show()

if __name__ == '__main__':
    graphic_sol(f2, (1, 0), (0, 100))