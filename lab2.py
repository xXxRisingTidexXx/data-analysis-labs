from typing import Optional
from numpy import array, log2, where, ndarray, sum


def main():
    p = array([
        [1 / 8, 1 / 16, 1 / 32, 1 / 32],
        [1 / 16, 1 / 8, 1 / 32, 1 / 32],
        [1 / 16, 1 / 16, 1 / 16, 1 / 16],
        [1 / 4, 0.0, 0.0, 0.0]
    ])
    ha = entropy(p, 1)
    hb = entropy(p, 0)
    hab = entropy(p)
    ha_b = conditional_entropy(p, 0)
    hb_a = conditional_entropy(p, 1)
    print(f'H(A) = {ha}')
    print(f'H(B) = {hb}')
    print(f'H(A,B) = {hab}')
    print(f'H(A|B) = {ha_b}')
    print(f'H(B|A) = {hb_a}')
    print(f'H(A,B) = {hab} = {ha + hb_a} = {hb + ha_b}')
    print(f'I(A,B) = I(B,A) = {ha - ha_b} = {hb - hb_a}')


def entropy(p: ndarray, axis: Optional[int] = None) -> float:
    """
    axis == 1, -> H(A)
    axis == 0, -> H(B)
    axis is None, -> H(A,B)
    """
    return sum(base_entropy(p if axis is None else sum(p, axis=axis)))  # noqa


def base_entropy(p: ndarray) -> ndarray:
    no_zeros = where(p == 0.0, 1.0, p)
    return -no_zeros * log2(no_zeros)


def conditional_entropy(p: ndarray, axis: int) -> float:
    """
    axis == 1, -> P(B|A)
    axis == 0, -> P(A|B)
    """
    return sum(
        sum(base_entropy(p / sum(p, axis=axis)), axis=axis) * sum(p, axis=axis)
    )  # noqa


if __name__ == '__main__':
    main()
