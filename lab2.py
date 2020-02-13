from typing import Tuple, Optional
from numpy import array, log2, where, ndarray, sum


def main():
    # pab = array([
    #     [0.1, 0.2, 0.25, 0.45],
    #     [0.05, 0.05, 0.3, 0.6],
    #     [0.05, 0.15, 0.25, 0.55],
    #     [0, 0.25, 0.35, 0.4]
    # ])
    p = array([
        [1 / 8, 1 / 16, 1 / 32, 1 / 32],
        [1 / 16, 1 / 8, 1 / 32, 1 / 32],
        [1 / 16, 1 / 16, 1 / 16, 1 / 16],
        [1 / 4, 0.0, 0.0, 0.0]
    ])
    print(f'H(A) = {entropy(p, 1)}')
    print(f'H(B) = {entropy(p, 0)}')
    print(f'H(A,B) = {entropy(p)}')


def entropy(p: ndarray, axis: Optional[int] = None) -> float:
    """
    axis == 1, -> H(A)
    axis == 0, -> H(B)
    axis is None, -> H(A,B)
    """
    total_p = p if axis is None else sum(p, axis=axis)
    no_zeros = where(total_p == 0.0, 1.0, total_p)
    return sum(-no_zeros * log2(no_zeros))  # noqa


if __name__ == '__main__':
    main()
