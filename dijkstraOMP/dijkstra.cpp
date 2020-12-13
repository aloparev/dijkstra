/** 
 * @brief: Dijkstra implementation with OpenMP
 * @author 557966
 * @date 31 V 2020
 *
 * OMP_NUM_THREADS=4 ./dipa
 * valgrind --time-stamp=yes --tool=helgrind --log-file=helgrind.log ./dipa
 * LD_LIBRARY_PATH=../../libgomp/build/x86_64-pc-linux-gnu/libgomp/.libs ./dipa
 */

#include <iostream>
#include <vector>
#include <string>
#include <list>

#include <set>
#include <utility> // for pair
#include <algorithm>
#include <iterator>

#include <ctime>
#include <time.h>
#include <omp.h>
#include <fstream>

#include "../utils/data.h"

/**
 * Given directed, weighted graph, compute shortest path 
 * \param source            source vertex as path start
 * \param adjacency_list    graph representation
 * \param min_distance      contains distances for each node
 * \param previous          contains the predecessor for each node
 */
void dijkstra(vertex_t source, const adjacency_list_t& adjacency_list, std::vector <weight_t>& min_distance, std::vector <vertex_t>& previous) {
    int graphSize = adjacency_list.size();

//    set weights on max, source on zero
    min_distance.clear();
    min_distance.resize(graphSize, max_weight);
    min_distance[source] = 0;

//    clear nodes, size excludes source
    previous.clear();
    previous.resize(graphSize, -1);

    std::set <std::pair<weight_t, vertex_t>> vertex_queue;
    vertex_queue.insert(std::make_pair(min_distance[source], source));

    omp_lock_t lock;
    omp_init_lock(&lock);

// nodes
    while (!vertex_queue.empty()) {
//        get key of [0]
        weight_t dist = vertex_queue.begin()->first;
        vertex_t u = vertex_queue.begin()->second;
        vertex_queue.erase(vertex_queue.begin());

        // Visit each edge exiting u
        
        const std::vector <neighbor>& neighbors = adjacency_list.find(u)->second;
        // const std::vector <neighbor>& neighbors = adjacency_list[u];
        int edgesSize = neighbors.size();

// neighbors
        #pragma omp parallel for //shared(vertex_queue, min_distance)
        for (int i=0; i < edgesSize; i++) {
            vertex_t v = neighbors[i].target;
            weight_t weight = neighbors[i].weight;
            weight_t distance_through_u = dist + weight;

            if (distance_through_u < min_distance[v]) {
                // #pragma omp critical 
                // { //writes
                    omp_set_lock(&lock);
                    // vertex_queue.erase(std::make_pair(min_distance[v], v));
                    min_distance[v] = distance_through_u;
                    previous[v] = u;
                    vertex_queue.insert(std::make_pair(min_distance[v], v));
                // }
                    omp_unset_lock(&lock);
            }
        }
        // #pragma omp barrier >> is here because of #parallel for
    }
    omp_destroy_lock(&lock);
}

/**
 * Launcher
 * \param argc      program argument counter
 * \param argv      submitted arguments: 1=graph file path, 2=source node, 3=target node
 * \return status   program exit status
 */
int main(int argc, char** argv) {
    clock_t begin_time = clock();
    const char* input_file_name;
    int start, end;

    printf("Dijkstra with OMP\nargc=%d\n", argc);
    if (argc == 4) {
        input_file_name = argv[1];
        start = atoi(argv[2]);
        end = (int) atoi(argv[3]);
        printf("init from args:\n\tinput file = %s\n\tsource = %d\n\ttarget = %d\n", 
            input_file_name, start, end);
    }
    else {
        input_file_name = "../resources/sampleGraph-1.gr";
        start = 0;
        end = 4;
        printf("no args submitted, use default values:\n\tinput file = %s\n\tsource = %d\n\ttarget = %d\n", input_file_name, start, end);
    }
    
    std::ifstream infile(input_file_name);
    adjacency_list_t adjacency_list; 
    readGR(infile, adjacency_list);
    // remember to insert edges both ways for an undirected graph
    // adjacency_list_t adjacency_list(6);
    std::vector <weight_t> min_distance;
    std::vector <vertex_t> previous;

    // printAdjacencyList(adjacency_list);
    printf("init in %f sec OK\n\n", float( clock () - begin_time ) /  CLOCKS_PER_SEC);

    begin_time = clock();
    dijkstra(start, adjacency_list, min_distance, previous);
    printf("distance from start node [%d] to end node [%d] is %.2f\ncalc time: %f sec\n",
        start, end, min_distance[end], float( clock () - begin_time ) /  CLOCKS_PER_SEC);

    std::list <vertex_t> path = getShortestPathToX(end, previous);
    std::cout << "path: ";
    std::copy(path.begin(), path.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}
