from typing import Callable
from numpy import array, ndarray, sum
from math import log2


def main():
    n, k = 8, 6
    a = generate(n, lambda i, j: i % j == 0 or j % i == 0)
    b = generate(n, lambda i, j: i == k or j == k)
    print(a)
    print(b)
    print(f'H(A) = {base_entropy(a)}')
    print(f'H(B) = {base_entropy(b)}')
    print(f'H(A|B) = {conditional_entropy(a, b)}')
    print(f'H(B|A) = {conditional_entropy(b, a)}')
    print(f'I(A,B) = {information(a, b):.10f} = I(B,A) = {information(b, a):.10f}')


def generate(size: int, generator: Callable) -> ndarray:
    return array([
        [int(generator(i, j)) for j in range(1, size + 1)]
        for i in range(1, size + 1)
    ])


def base_entropy(a: ndarray) -> float:
    return entropy(probability(a))


def probability(a: ndarray) -> float:
    return sum(a) / len(a) / len(a)


def entropy(p: float) -> float:
    return 0 if p == 0 or p == 1 else -p * log2(p) + (p - 1) * log2(1 - p)


def conditional_entropy(a: ndarray, b: ndarray) -> float:
    nb = negate(b)
    return (
        probability(b) * entropy(conditional_probability(a, b))
        + probability(nb) * entropy(conditional_probability(a, nb))
    )


def negate(a: ndarray) -> ndarray:
    return (a + 1) % 2


def conditional_probability(a: ndarray, b: ndarray) -> float:
    return sum(a * b) / sum(b)


def information(a: ndarray, b: ndarray) -> float:
    return base_entropy(a) - conditional_entropy(a, b)


if __name__ == '__main__':
    main()
