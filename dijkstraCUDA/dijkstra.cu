#include <iostream>
#include <vector>
#include <string>
#include <list>

// #include <limits> // for numeric_limits

#include <set>
#include <utility> // for pair
#include <algorithm>
#include <iterator>

#include <ctime>
#include <time.h>
#include <fstream>

// #include "data.h"
#include "../utils/data.h"

/*
nvcc ./app.cu –o app
./app
https://stackoverflow.com/questions/18963293/cuda-atomics-change-flag/18968893#18968893
*/

__global__ void checkNeighbors(const int* matrix, const int size, const int u, int* min_distance_dev, int* previous_dev) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid < size) {
        if(tid == u || matrix[u*size + tid] == -1) continue;

        int w = matrix[u*size + tid];
            weight_t distance_through_u = w + min_distance_dev[u];

            if (distance_through_u < min_distance[tid]) {
                atomicMin(&min_distance_dev[tid], distance_through_u);
                atomicMax(&previous_dev[tid], u);
                    // min_distance[v] = distance_through_u;   //atomic min
                    // previous[v] = u;                        //atomic max
                    // printf("relaxation: u=%d ud=%.2f v=%d vd=%.2f\n", u+1, dist, v+1, distance_through_u);
            }

        }
    }

void dijkstra(const vertex_t source, const std::vector<int>& matrix, const int size, std::vector <weight_t>& min_distance, std::vector <vertex_t>& previous) {
    // int size = static_cast<int>(matrix.size());

//    set weights on max, source on zero
    min_distance.clear();
    min_distance.resize(size, max_weight);
    min_distance[source] = 0;

//    clear nodes, size excludes source
    previous.clear();
    previous.resize(size, -1);

    std::vector<bool> visited;
    visited.resize(size, false);

    int* min_distance_dev;
    int* previous_dev;
    int* matrix_dev;

    cudaMalloc( &min_distance_dev, size*sizeof(double) );
    cudaMalloc( &previous_dev, size*sizeof(int) );
    cudaMalloc( &matrix_dev, size*size*sizeof(int) );

    cudaMemcpy( matrix_dev, matrix.data(), size*size*sizeof(double), cudaMemcpyHostToDevice );

    /*copy to device
        matrix
        dist
        prev
        fixed u
    */

    while (true) {
        weight_t dist = max_weight;
        vertex_t u = -1;

        for (int i=0; i < size*size; i+=size) {
            if(!visited[i/size] && min_distance[i/size] < dist) {
                // dist = min_distance[i];
                u = i/size;
            }
        }
        
        if(u == -1) { break;} //exit, if no unvisited nodes  
        visited[u] = true;  

        cudaMemcpy( min_distance_dev, min_distance.data(), size*sizeof(double), cudaMemcpyHostToDevice );
        cudaMemcpy( previous_dev, previous.data(), size*sizeof(int), cudaMemcpyHostToDevice );

        checkNeighbors <<< 1, size >>> (matrix, size, u, min_distance_dev, previous_dev);

        cudaMemcpy( min_distance.data(), min_distance_dev, size*sizeof(double), cudaMemcpyDeviceToHost );
        cudaMemcpy( previous.data(), previous_dev, size*sizeof(int), cudaMemcpyDeviceToHost );
    }   
}

std::list <vertex_t> getShortestPathToX(vertex_t vertex, const std::vector <vertex_t>& previous) {
    std::list <vertex_t> path;
    
     do {
        path.push_front(vertex+1);
        vertex = previous[vertex];
     } while(vertex != -1);
    return path;
}

int main() {
    std::ifstream infile("../resources/sampleGraph-1.gr");
    int size;
    std::vector<int> matrix;

    readGR4(infile, matrix, size);
    printAdjacencyMatrix2(matrix, size);

    std::vector <weight_t> min_distance;
    std::vector <vertex_t> previous;
    int start = 0;
    int target = 4;

    std::printf("init OK\n\n");

    const clock_t begin_time = clock();
    dijkstra(start, matrix, size, min_distance, previous);
    std::printf("distance from start node [%d] to end node [%d] is %.2f\n"
                "calculation time: %f sec\n",
                start, target, min_distance[target], float( clock () - begin_time ) /  CLOCKS_PER_SEC);

    std::list <vertex_t> path = getShortestPathToX(target, previous);
    std::cout << "path: ";
    std::copy(path.begin(), path.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}
