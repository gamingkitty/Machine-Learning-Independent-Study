import math


def sub(arr1, arr2):
    return [arr1[i] - arr2[i] for i in range(len(arr1))]


def add(arr1, arr2):
    return [arr1[i] + arr2[i] for i in range(len(arr1))]


def mult(arr1, arr2):
    return [arr1[i] * arr2[i] for i in range(len(arr1))]


def divide(arr1, arr2):
    return [arr1[i] / arr2[i] for i in range(len(arr1))]


def dot(arr1, arr2):
    return sum(arr1[i] * arr2[i] for i in range(len(arr1)))

def matrix_mult(arr1, arr2):
    if not is_2d(arr1):
        arr1 = [arr1]
    if not is_2d(arr2):
        arr2 = [arr2]
    return [[dot(row, col) for col in transpose(arr2)] for row in arr1]

def transpose(arr):
    if not is_2d(arr):
        arr = [arr]
    return [[row[i] for row in arr] for i in range(len(arr[0]))]

def is_2d(arr):
    return isinstance(arr[0], list)

def mult_scalar(arr, scalar):
    return [scalar * data for data in arr]


def ln(arr):
    return [math.log(num) for num in arr]


def exp(arr):
    return [math.exp(num) for num in arr]


def project(point, base):
    return mult_scalar(base, proj_mult(point, base))


def proj_mult(point, base):
    return dot(point, base) / dot(base, base)


def ones(n):
    return [1 for _ in range(n)]


def zeros(n):
    return [0 for _ in range(n)]


def power(vector, n):
    return [(p ** n) for p in vector]


def dist(p1, p2):
    return math.sqrt(sum(power(sub(p1, p2), 2)))
