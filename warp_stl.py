from reader import read_stl_verticies
from utils import split_triangles, push2d, top_wider, push3d
from writer import Binary_STL_Writer

if __name__ == '__main__':
    faces = []
    triangles = read_stl_verticies("./input.stl")
    more_triangles = split_triangles(triangles,5)
    for triangle in more_triangles:
        triangle = list(triangle)
        for i in range(3):
            #Edit to change how the STL is warped
            triangle[i] = push3d(triangle[i],[0,0,10],400)
            triangle[i] = top_wider(triangle[i],2)
        faces.append(triangle)
    with open('output.stl', 'wb') as fp:
        writer = Binary_STL_Writer(fp)
        writer.add_faces(faces)
        writer.close()
