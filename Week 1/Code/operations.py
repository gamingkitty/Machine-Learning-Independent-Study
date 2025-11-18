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


def mult_scalar(arr, scalar):
    return [scalar * data for data in arr]

def max(arr):
    return max(arr)

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
