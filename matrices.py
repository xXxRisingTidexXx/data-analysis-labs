from numpy import array
from numpy.linalg import inv


def main():
    size = 5
    a = array([
        [8.0, 4.0, 9.0, 8.0, 7.0],
        [8.0, 6.0, 5.0, 9.0, 7.0],
        [3.0, 9.0, 8.0, 7.0, 7.0],
        [8.0, 9.0, 3.0, 2.0, 5.0],
        [6.0, 3.0, 5.0, 7.0, 8.0]
    ])
    q = array([199.0, 181.0, 187.0, 142.0, 152.0])
    augmented = array([
        [8.0, 4.0, 9.0, 8.0, 7.0, 199.0],
        [8.0, 6.0, 5.0, 9.0, 7.0, 181.0],
        [3.0, 9.0, 8.0, 7.0, 7.0, 187.0],
        [8.0, 9.0, 3.0, 2.0, 5.0, 142.0],
        [6.0, 3.0, 5.0, 7.0, 8.0, 152.0]
    ])
    for i in range(size):
        for j in range(i + 1, size):
            augmented[j] -= (augmented[j, i] / augmented[i, i]) * augmented[i]
        augmented[i] /= augmented[i, i]
    for i in range(size - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            augmented[j] -= (augmented[j, i] / augmented[i, i]) * augmented[i]
    print(augmented[:, size])
    print(inv(a) @ q)


if __name__ == '__main__':
    main()
