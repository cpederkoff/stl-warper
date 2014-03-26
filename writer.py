#!/usr/bin/env python
#coding:utf-8
# Purpose: Export 3D objects, build of faces with 3 or 4 vertices, as ASCII or Binary STL file.
# License: MIT License
from itertools import combinations
import math
import numpy as np
import struct
from reader import readtardis

ASCII_FACET = """facet normal 0 0 0
outer loop
vertex {face[0][0]:.4f} {face[0][1]:.4f} {face[0][2]:.4f}
vertex {face[1][0]:.4f} {face[1][1]:.4f} {face[1][2]:.4f}
vertex {face[2][0]:.4f} {face[2][1]:.4f} {face[2][2]:.4f}
endloop
endfacet
"""

BINARY_HEADER ="80sI"
BINARY_FACET = "12fH"

class ASCII_STL_Writer:
    """ Export 3D objects build of 3 or 4 vertices as ASCII STL file.
    """
    def __init__(self, stream):
        self.fp = stream
        self._write_header()

    def _write_header(self):
        self.fp.write("solid python\n")

    def close(self):
        self.fp.write("endsolid python\n")

    def _write(self, face):
        self.fp.write(ASCII_FACET.format(face=face))

    def _split(self, face):
        p1, p2, p3, p4 = face
        return (p1, p2, p3), (p3, p4, p1)

    def add_face(self, face):
        """ Add one face with 3 or 4 vertices. """
        if len(face) == 4:
            face1, face2 = self._split(face)
            self._write(face1)
            self._write(face2)
        elif len(face) == 3:
            self._write(face)
        else:
            raise ValueError('only 3 or 4 vertices for each face')

    def add_faces(self, faces):
        """ Add many faces. """
        for face in faces:
            self.add_face(face)

class Binary_STL_Writer(ASCII_STL_Writer):
    """ Export 3D objects build of 3 or 4 vertices as binary STL file.
    """
    def __init__(self, stream):
        self.counter = 0
        super(Binary_STL_Writer, self).__init__(stream)

    def close(self):
        self._write_header()

    def _write_header(self):
        self.fp.seek(0)
        self.fp.write(struct.pack(BINARY_HEADER, b'Python Binary STL Writer', self.counter))

    def _write(self, face):
        self.counter += 1
        data = [
            0., 0., 0.,
            face[0][0], face[0][1], face[0][2],
            face[1][0], face[1][1], face[1][2],
            face[2][0], face[2][1], face[2][2],
            0
        ]
        self.fp.write(struct.pack(BINARY_FACET, *data))



def push(point,center):
    distancevals = list(p-c  for p,c in zip(point,center))
    distance = (distancevals[0]**2 + distancevals[1]**2)**(1/2)
    multiplier = 20/(distance+1)
    return [point[0]+point[0]*multiplier,point[1]+point[1]*multiplier,point[2]]

def topwider(point):
    multiplier = point[2]/200+1
    return [point[0]*multiplier, point[1]*multiplier, point[2]]
#determinant of matrix a

def dist(p1,p2):
    return ((p1[0]-p2[0])**2 +
            (p1[1]-p2[1])**2 +
            (p1[2]-p2[2])**2)**.5

def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x,y,z = normal((a,b,c))
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

def normal(triangle):
    a = triangle[0]
    b=triangle[1]
    c=triangle[2]
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    return (x,y,z)
def magnatude(triangle):
    s = 0
    for point in triangle:
        s+=point**2
    return s**.5

#dot product of vectors a and b
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def angle(v1, v2):
  return (math.acos(dot(v1, v2) / (magnatude(v1) * magnatude(v2)+.000000001)))/math.pi*180

#cross product of vectors a and b
def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)

#area of polygon poly
def area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)

        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def splittriangles(triangles):
    for triangle in triangles:
        for split in splittriangle(triangle):
            yield split

def splittriangle(triangle):
    ar = area(triangle)
    if ar < 2 :
        return [triangle]
    else:
        pairs = list(combinations(triangle,2))
        longpair = max(pairs,key=lambda pair:dist(pair[0],pair[1]))
        mid = np.array(list((p1+p2)/float(2.0) for p1,p2 in zip(longpair[0],longpair[1])),dtype=np.float32)
        for point in triangle:
            seen = False
            for longpoint in longpair:
                if np.array_equal(point,longpoint):
                    seen = True
            if not seen:
                oddpoint = point
        oddpoint = np.array(oddpoint,dtype=np.float32)


        t1 = [oddpoint,mid,longpair[1]]
        t2 = [mid,oddpoint,longpair[0]]
        if angle(normal(triangle), normal(t1))>90:
            t1.reverse()
        if angle(normal(triangle), normal(t2))>90:
            t2.reverse()
        ts = []
        ts.extend(splittriangle(t1))
        ts.extend(splittriangle(t2))
        return ts

if __name__ == '__main__':
    faces = []
    triangles = readtardis()
    moretriangles = splittriangles(triangles)
    for triangle in moretriangles:
        c1 = [0,0,0]
        triangle = list(triangle)

        for i in range(3):
            triangle[i] = push(triangle[i],c1)
        for i in range(3):
            triangle[i] = topwider(triangle[i])
        faces.append(triangle)
    with open('cube.stl', 'wb') as fp:
        writer = Binary_STL_Writer(fp)
        writer.add_faces(faces)
        writer.close()

