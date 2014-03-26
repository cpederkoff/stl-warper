import math
import numpy as np
from itertools import combinations

#pushes a point away from the center. Only applies force in 2d. Force does not change in z.
def push2d(point, center, amount):
    distance_vals = list(p - c for p, c in zip(point, center))
    distance = (distance_vals[0] ** 2 + distance_vals[1] ** 2) ** (1 / 2)
    multiplier = amount / (distance + 1)
    return [point[0] + point[0] * multiplier, point[1] + point[1] * multiplier, point[2]]

#pushes a point away from the center. Points can be pushed in z, and force changes in z.
def push3d(point, center, amount):
    distance_vals = list(p - c for p, c in zip(point, center))
    distance = (distance_vals[0] ** 2 + distance_vals[1] ** 2 + distance_vals[2] ** 2) ** (1 / 2)
    multiplier = (amount / (distance + .00000001)**2)+1
    return [point[0] * multiplier, point[1] * multiplier, point[2]*multiplier]

#Pushes a point away from the center depending on how high it is.
def top_wider(point, amount):
    amount = (point[2] * amount / 200) + 1
    return [point[0] * amount, point[1] * amount, point[2]]


#determinant of matrix a
def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2) ** .5


def det(a):
    return a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1] - a[0][2] * a[1][1] * \
           a[2][0] - a[0][1] * a[1][0] * a[2][2] - a[0][0] * a[1][2] * a[2][1]


#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x, y, z = normal((a, b, c))
    magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
    return (x / magnitude, y / magnitude, z / magnitude)


def normal(triangle):
    a = triangle[0]
    b = triangle[1]
    c = triangle[2]
    x = det([[1, a[1], a[2]],
             [1, b[1], b[2]],
             [1, c[1], c[2]]])
    y = det([[a[0], 1, a[2]],
             [b[0], 1, b[2]],
             [c[0], 1, c[2]]])
    z = det([[a[0], a[1], 1],
             [b[0], b[1], 1],
             [c[0], c[1], 1]])
    return (x, y, z)


def magnitude(triangle):
    s = 0
    for point in triangle:
        s += point ** 2
    return s ** .5


#dot product of vectors a and b
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def angle(v1, v2):
    return (math.acos(dot(v1, v2) / (magnitude(v1) * magnitude(v2) + .000000001))) / math.pi * 180


#cross product of vectors a and b
def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)


#area of polygon poly
def area(poly):
    if len(poly) < 3:  # not a plane - no area
        return 0
    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly) - 1:
            vi2 = poly[0]
        else:
            vi2 = poly[i + 1]
        prod = cross(vi1, vi2)

        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result / 2)


def split_triangles(triangles,max_area):
    for triangle in triangles:
        for split in split_triangle(triangle, max_area):
            yield split


def split_triangle(triangle, max_area):
    ar = area(triangle)
    if ar < max_area:
        yield triangle
    else:
        #
        sides = list(combinations(triangle, 2))
        #find the longest side of the triangle
        longest_side = max(sides, key=lambda side: dist(side[0], side[1]))
        midpoint = np.array(list((p1 + p2) / float(2.0) for p1, p2 in zip(longest_side[0], longest_side[1])), dtype=np.float32)
        #Ugly because couldn't find better way of telling membership in an array of an array
        for point in triangle:
            seen = False
            for long_point in longest_side:
                if np.array_equal(point, long_point):
                    seen = True
            if not seen:
                opposite_point = point
        #opposite_point is the point that is opposite the longest side
        opposite_point = np.array(opposite_point, dtype=np.float32)

        t1 = [opposite_point, midpoint, longest_side[1]]
        t2 = [midpoint, opposite_point, longest_side[0]]
        #if the angle is > 90 it means that the normals are not pointing the same direction
        if angle(normal(triangle), normal(t1)) > 90:
            #reverses the normal, due to the right hand rule
            t1.reverse()
        if angle(normal(triangle), normal(t2)) > 90:
            t2.reverse()
        for t in split_triangle(t1, max_area):
            yield t
        for t in split_triangle(t2, max_area):
            yield t