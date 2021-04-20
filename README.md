This repository contains two mini-apps that were created to improve the MOAB remapping workflow for Climate problems.

1) Kd-tree mini-app implementation: 
   This example loads a 2d/3d unstructured mesh file and performs several queries to locate the points belonging to elements in the mesh.
   
   Typical usage:
        `./kdtreeApp -i MESH_FILE -d 3 -n 100000 -r 1`
   
   There are parameters that can be specified for the application.
     a) `-i <path>` - Path to the mesh file to load and query.
     b) `-d <int>` - Leading dimension of the mesh file. Default is 3.
     c) `-n <int>` - Number of queries to perform on the tree datastructure. Default is 1000000.
     c) `-r <int>` - Number of in-memory refinements to perform to have a much bigger mesh to query. Default is 0.

2) SpMV(n) mini-app implementation: 
   This example loads a remapping weight matrix from file, and forms a SparseMatrix datastructure in memory. Then it applies the matrix onto a random vector multiple times to compute average time for SpMV. 

   Typical usage follows:
        `./spmvApp OPERATOR_PATH -t -v 1 -n 100`
   
   There are parameters that can be varied here.
     a) `OPERATOR_PATH` - This is a required argument specifying the path to the NetCDF-based matrix weights file.
     a) `-t` - Enable both A*x and A^T*x operations in timing profile. Default is False.
     b) `-v <int>` - This argument controls the number of vectors onto which the operator is applied. This is useful to mimic transfer of multiple real world fields between components. Default is 1.
     c) `-n <int>` - This number controls the total iterations to perform to get an average SpMV timing. Default is 100.

