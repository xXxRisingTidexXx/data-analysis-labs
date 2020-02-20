from numpy import array
from numpy.linalg import det

zero = array([
    [1.56, 1.56, 1.56],
    [1.24, 1.24, 1.24],
    [0.153, 0.153, 0.153]
])
vertices = array([
    [0.79, 3.95, 2.4],
    [3.48, 3.9, 2.46],
    [0.138, 0.141, 0.072]
])
volume = abs(det(vertices - zero))
print(f'V = {volume:.6f}')
print(f'm = {volume * 2.711:.6f}')

points = [
    [4484, 2963],
    [4444, 3363],
    [1852, 4876],
    [560, 2342],
    [3094, 1050]
]
areas = [
    abs(
        (points[i - 1][0] - points[0][0]) * (points[i][1] - points[0][1])
        - (points[i][0] - points[0][0]) * (points[i - 1][1] - points[0][1])
    ) / 2
    for i in range(2, len(points))
]
print(f'S = {sum(areas)}')
