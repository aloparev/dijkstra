# shortest path parallel


## requirements

- own libgomp build
- include custom omp.h
- big graph files
- CUDA environment


## getting started

- ```make``` will create executables in every folder, which can be run
- in case of CUDA, however, we need to copy the files to device first
- run parameters for graph file, start and end nodes are accepted 
    - e.g. ```dijkstra/dijkstra resources/ny-roads.gr 0 25907```
