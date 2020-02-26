from numpy import array, reshape, hstack, sum, round
from math import sqrt
from numpy.linalg import inv


def main():
    size = 5
    a = array([
        [35.0, 51.0, 30.0, 43.0, 1.0],
        [7.0, 31.0, 1.0, 19.0, 31.0],
        [32.0, 48.0, 3.0, 25.0, 51.0],
        [61.0, 39.0, 9.0, 35.0, 15.0],
        [43.0, 34.0, 25.0, 22.0, 13.0]
    ])
    q = array([4274.0, 2953.0, 4875.0, 3507.0, 3661.0])
    augmented = hstack((a, reshape(q, (size, 1))))
    for i in range(size):
        for j in range(i + 1, size):
            augmented[j] -= (augmented[j, i] / augmented[i, i]) * augmented[i]
        augmented[i] /= augmented[i, i]
    for i in range(size - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            augmented[j] -= (augmented[j, i] / augmented[i, i]) * augmented[i]
    a6 = array([15.0, 1.0, 9.0, 51.0, 33.0])
    a7 = array([56.0, 29.0, 2.0, 46.0, 19.0])
    # print(inv(a) @ q)
    w = augmented[:, size]
    print(sum(a6 * w))
    print(sum(a7 * w))
    y = round(w)
    print(w)
    print(abs(sqrt(sum(w * w)) - sqrt(sum(y * y))))


if __name__ == '__main__':
    main()
