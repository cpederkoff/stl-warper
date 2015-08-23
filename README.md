# stl-warper
Apply forces, push and pull verticies in an STL file.
### Main Features
* Apply a force or formula on every vertex of a given mesh
* Output back to STL
* Command line interface

### How to use
* Open warp_stl.py, and edit the line that says:
```
#Edit to change how the STL is warped
```
* Add functions to return new coordinates depending on where the point is. 
* Edit the input.stl and output.stl strings to reflect the actual locations of files

### Example: 
![alt text](https://github.com/rcpedersen/stl-warper/raw/master/pictures/tardis.png "Original model")
![alt text](https://github.com/rcpedersen/stl-warper/raw/master/pictures/tardis_warped.png "Model after warping")
