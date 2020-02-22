from functools import reduce
from typing import List
from numpy import ndarray, array, log2, sum, where
from numpy.random import choice
from math import factorial, log2 as lb


def main():
    x = choice(6, 25, p=[0.05, 0.07, 0.2, 0.3, 0.35, 0.03])
    print(x)
    print(f'Hc = {combination_entropy(x, 6)}')
    print(f'H = {entropy(x, 6)}')


def combination_entropy(x: ndarray, size: int) -> float:
    return (
        lb(
            reduce(
                lambda a, c: a / factorial(c),
                count(x, size),
                factorial(len(x))
            )
        )
        / len(x)
    )


def count(x: ndarray, size: int) -> List[int]:
    counts = [0] * size
    for xi in x:
        counts[xi] += 1
    return counts


def entropy(x: ndarray, size: int) -> float:
    p = array(count(x, size)) / len(x)
    no_zeros = where(p == 0.0, 1.0, p)
    return sum(-no_zeros * log2(no_zeros))  # noqa


if __name__ == '__main__':
    main()
