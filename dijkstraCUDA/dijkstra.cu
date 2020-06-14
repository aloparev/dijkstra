#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>

#include "data.h"

/*
nvcc ./app.cu –o app
./app
*/

__global__
void gpuEdgeCheck() {
	__shared__std::vector <weight_t> min_distance;
	__shared__std::vector <vertex_t> previous;
	__shared__std::set <std::pair<weight_t, vertex_t>> vertex_queue;

	int i = threadIdx.x;
	vertex_t v = neighbors[i].target;
    weight_t weight = neighbors[i].weight;
    weight_t distance_through_u = dist + weight;
    __syncthreads();

    //how to sync access to queue?
}

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
                // #pragma omp critical { //writes
                    omp_set_lock(&lock);
                    vertex_queue.erase(std::make_pair(min_distance[v], v));
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

int main() {
    std::ifstream infile("../resources/sampleGraph-1.gr");
    adjacency_list_t adjacency_list; 
    readGR(infile, adjacency_list);
    std::vector <weight_t> min_distance;
    std::vector <vertex_t> previous;
    int start = 1;
    int target = 5;

    printAdjacencyList(adjacency_list);
    std::printf("init OK\n\n");

    // const clock_t begin_time = clock();
    // dijkstrap(start, adjacency_list, min_distance, previous);
    // std::printf("distance from start node [%d] to end node [%d] is %.2f\n"
    //             "calculation time: %f sec\n",
    //             start, target, min_distance[target], float( clock () - begin_time ) /  CLOCKS_PER_SEC);

    // std::list <vertex_t> path = getShortestPathToX(target, previous);
    // std::cout << "Path : ";
    // std::copy(path.begin(), path.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
    // std::cout << std::endl;

    return 0;
}
