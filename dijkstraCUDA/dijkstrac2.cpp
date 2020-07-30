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

/** @brief Dijkstra implementation with OpenMP
    @author 557966
    @date 31 V 2020
    doxygen test
*/

/* 
run: 
    OMP_NUM_THREADS=4 ./dipa
    valgrind --time-stamp=yes --tool=helgrind --log-file=helgrind.log ./dipa

    LD_LIBRARY_PATH=../../libgomp/build/x86_64-pc-linux-gnu/libgomp/.libs ./dipa
*/

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
                dist = min_distance[i];
                u = i/size;
            }
        }
        
        if(u == -1) { break;} //exit, if no unvisited nodes  
        visited[u] = true;  

        //call kernel with threads=neighbors
        for(int v=0; v<size; v++) {
            if(u == v || matrix[u*size + v] == -1) {
                printf("continue: u=%d ud=%.2f v=%d vd=%.2f\n", u, dist, v, dist + matrix[u*size+v]);
                continue;
            }

            int w = matrix[u*size + v];
            weight_t distance_through_u = w + min_distance[u];

            if (distance_through_u < min_distance[v]) {
                    min_distance[v] = distance_through_u;   //atomic min
                    previous[v] = u;                        //atomic max
                    printf("relaxation: u=%d ud=%.2f v=%d vd=%.2f\n", u, dist, v, distance_through_u);
            }

        }

        // for(int v=0; v<size; v++) {
        //     printf("node=%d d=%.2f v=%s p=%d\n", v+1, min_distance[v], visited[v] ? "true" : "false", previous[v]+1);
        // }
        // std::cout << std::endl;

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
