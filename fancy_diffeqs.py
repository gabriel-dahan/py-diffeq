# Under MIT Licence - Gabriel D. | See LICENCE.md

from scipy.integrate import odeint
import numpy as np

import matplotlib.pyplot as plt
import shutil

from fractions import Fraction

## --- Graphics --- ##

def fancy_exp(n: int) -> str:
    exps = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
    chr_ = str(n)
    return ''.join(exps[int(e)] for e in chr_)

def sign(n: int) -> str:
    return '+' if n >= 0 else '-'

## ---------------- ##

def has_latex() -> bool:
    return True if shutil.which('latex') else False

LATEX = has_latex()

plt.rcParams['text.usetex'] = LATEX

## ---------------- ##


class LinearDiffEq:

    """ 
        Operations on Differential equations.
        The coefficients of the differential equation are entered in increasing order of the derivatives.
    """

    def __init__(self, init: tuple[int], *coeff: tuple[int | Fraction]) -> None:
        self._init = init
        self._coeffs = np.array(coeff)
        self.order = self._coeffs.shape[0] - 2
        assert len(init) == self.order, \
            'The number of initial conditions for the differential equation must be equal to its order.'
        
        self.canonize()
        
    def graph(self, range_: tuple = (0, 150), n = 150) -> None:
        t, x = self.solve(range_, n)

        plt.plot(t, x)
        plt.xlabel(rf'$t\\in\\[{range_[0]},{range_[1]}\\]$' if LATEX else f't from {range_[0]} to {range_[1]}')
        plt.ylabel(rf'Sol. {"$y$" if LATEX else "y"}')
        plt.title(rf'Solution de {self.__latex_str__()}' if LATEX else f'Solution de {self}')
        plt.grid()

        plt.show()

    def __order1(self, t: np.ndarray) -> tuple[list[float]]:
        return t, []

    def solve(self, range_: tuple, n: int = 150) -> tuple[list[float]]:
        t = np.linspace(range_[0], range_[1], n)
        if self.order == 0:
            return t, np.full(t.shape, -self._coeffs[0])
        if self.order == 1:
            return self.__order1(t)
        sol = odeint(self.__system, self._init, t)
        return t, sol[:, 0]

    def canonize(self) -> None:
        self._coeffs = np.array([Fraction(coeff, self._coeffs[-1]) for coeff in self._coeffs])

    def __system(self, X: list[int], t: int) -> list[int]:
        """ Characteristic function of odeint. """
        C = -self._coeffs
        return [X[i] for i in range(1, self.order)] \
                + [C[0] + sum(C[i] * X[i - 1] for i in range(1, self.order + 1))]

    def __repr__(self) -> str:
        return f"({', '.join(str(c) for c in self._coeffs)})"
    
    def __str__(self) -> str:
        return ''.join(f'{sign(c)}{abs(c)}{"y⁽" + fancy_exp(self.order - i) + "⁾" if self.order - i + 1 > 0 else ""}' \
                       for i, c in enumerate(np.flipud(self._coeffs))) + ' = 0'
    def __latex_str__(self) -> str:
        return ''.join(f'{sign(c)}{abs(c)}{"y⁽" + fancy_exp(self.order - i) + "⁾" if self.order - i + 1 > 0 else ""}' \
                       for i, c in enumerate(np.flipud(self._coeffs))) + ' = 0'

    def __eq__(self, __value: 'LinearDiffEq') -> bool:
        return __value._coeffs == self._coeffs
    
    def __add__(self, __value: 'LinearDiffEq') -> 'LinearDiffEq':
        i1, i2 = __value._init, self._init
        assert i1 == i2, f'The initial conditions must be the same when adding differential equations : {i1} != {i2}.'
        return LinearDiffEq(self._init, *(__value._coeffs + self._coeffs))

if __name__ == '__main__':
    init = (1, 0)
    coeffs = [-1, 8, 1, 2]
    de = LinearDiffEq(init, *coeffs)
    print(de + LinearDiffEq((0, 1), -1, 4, -7, 2)) # --> Error
    # de.graph((0, 200), 1500)