#include <iostream>
#include <vector>
#include <string>
#include <list>

#include <limits> // for numeric_limits

#include <set>
#include <utility> // for pair
#include <algorithm>
#include <iterator>

#include <ctime>
#include <time.h>
#include <omp.h>

/* 
to switch between sequential or parallel version call the corresponding function from main
    dijkstra() is seq.
    dijkstrap() is parallel

sequential compile: 
    g++ dijkstra.cpp -o dijkstra
run: ./dijkstra

parallel compile: 
    g++ -fopenmp dijkstra.cpp -o dijkstra2 -g
run: 
    OMP_NUM_THREADS=4 ./dijkstra2
    valgrind --time-stamp=yes --tool=helgrind --log-file=helgrind.log ./dijkstra2
*/

// self defined types to quickly change type if need be
typedef int vertex_t; // vertex=node
typedef double weight_t;

struct neighbor {
    vertex_t target;
    weight_t weight;

    neighbor(vertex_t arg_target, weight_t arg_weight)
            : target(arg_target), weight(arg_weight) {}
};

typedef std::vector <std::vector<neighbor>> adjacency_list_t;
const weight_t max_weight = std::numeric_limits<double>::infinity();

// sequential version
void dijkstra(vertex_t source,
//                          references same memory cells
                          const adjacency_list_t &adjacency_list,
                          std::vector <weight_t> &min_distance,
                          std::vector <vertex_t> &previous) {
    int n = adjacency_list.size();

    min_distance.clear();

//    set weights on max, source on zero
    min_distance.resize(n, max_weight);
    min_distance[source] = 0;

//    clear nodes, size excludes source
    previous.clear();
    previous.resize(n, -1);

    std::set <std::pair<weight_t, vertex_t>> vertex_queue;
    vertex_queue.insert(std::make_pair(min_distance[source], source));

// nodes
    while (!vertex_queue.empty()) {

//        get key of [0]
        weight_t dist = vertex_queue.begin()->first;
        vertex_t u = vertex_queue.begin()->second;
        vertex_queue.erase(vertex_queue.begin());

        // Visit each edge exiting u
        const std::vector <neighbor> &neighbors = adjacency_list[u];
        int nSize = neighbors.size();

// neighbors
        for (int i = 0; i < nSize; i++) {
            vertex_t v = neighbors[i].target;
            weight_t weight = neighbors[i].weight;
            weight_t distance_through_u = dist + weight;

            if (distance_through_u < min_distance[v]) {
                vertex_queue.erase(std::make_pair(min_distance[v], v));

                min_distance[v] = distance_through_u;
                previous[v] = u;
                vertex_queue.insert(std::make_pair(min_distance[v], v));
            }
        }
    }
}

// parallel version
void dijkstrap(vertex_t source,
//                          references same memory cells
                          const adjacency_list_t &adjacency_list,
                          std::vector <weight_t> &min_distance,
                          std::vector <vertex_t> &previous) {
    int n = adjacency_list.size();

//    set weights on max, source on zero
    min_distance.clear();
    min_distance.resize(n, max_weight);
    min_distance[source] = 0;

//    clear nodes, size excludes source
    previous.clear();
    previous.resize(n, -1);

    std::set <std::pair<weight_t, vertex_t>> vertex_queue;
    vertex_queue.insert(std::make_pair(min_distance[source], source));

    // omp_lock_t lock;
    // omp_init_lock(&lock);

// nodes
    while (!vertex_queue.empty()) {

//        get key of [0]
        weight_t dist = vertex_queue.begin()->first;
        vertex_t u = vertex_queue.begin()->second;
        vertex_queue.erase(vertex_queue.begin());

        // Visit each edge exiting u
        
        const std::vector <neighbor> &neighbors = adjacency_list[u];
        int nSize = neighbors.size();

// neighbors
        #pragma omp parallel for //shared(vertex_queue, min_distance)
        for (int i = 0; i < nSize; i++) {
            vertex_t v = neighbors[i].target;
            weight_t weight = neighbors[i].weight;
            weight_t distance_through_u = dist + weight;

            if (distance_through_u < min_distance[v]) {
                #pragma omp critical //writes
                {
                    // omp_set_lock(&lock);
                    vertex_queue.erase(std::make_pair(min_distance[v], v));
                    min_distance[v] = distance_through_u;
                    previous[v] = u;
                    vertex_queue.insert(std::make_pair(min_distance[v], v));
                }
                    // omp_unset_lock(&lock);
            }
        }
        // #pragma omp barrier >> is here because of for loop parallel
    }
    // omp_destroy_lock(&lock);
}

std::list <vertex_t> getShortestPathToX(vertex_t vertex, const std::vector <vertex_t> &previous) {
    std::list <vertex_t> path;
    
     do {
        path.push_front(vertex);
        vertex = previous[vertex];
     } while(vertex != -1);
    return path;
}

void printAdjacencyList(const adjacency_list_t &adjacency_list) {
    int size = adjacency_list.size();
    std::printf("printing graph\n");

    for(int i=0; i<size; i++) {
        std::printf("node[%d]: ", i);

        for(auto const& value: adjacency_list[i]) {
            std::printf("(%d, %.2f) ", value.target, value.weight);
        }
        std::printf("\n");
    }
        // std::printf("\n");
}

int main() {
    // remember to insert edges both ways for an undirected graph
    adjacency_list_t adjacency_list(6);
    int start = 0;
    int target = 4;

    // 0 = a
    adjacency_list[0].push_back(neighbor(1, 7));
    adjacency_list[0].push_back(neighbor(2, 9));
    adjacency_list[0].push_back(neighbor(5, 14));
    // 1 = b
    adjacency_list[1].push_back(neighbor(0, 7));
    adjacency_list[1].push_back(neighbor(2, 10));
    adjacency_list[1].push_back(neighbor(3, 15));
    // 2 = c
    adjacency_list[2].push_back(neighbor(0, 9));
    adjacency_list[2].push_back(neighbor(1, 10));
    adjacency_list[2].push_back(neighbor(3, 11));
    adjacency_list[2].push_back(neighbor(5, 2));
    // 3 = d
    adjacency_list[3].push_back(neighbor(1, 15));
    adjacency_list[3].push_back(neighbor(2, 11));
    adjacency_list[3].push_back(neighbor(4, 6));
    // 4 = e
    adjacency_list[4].push_back(neighbor(3, 6));
    adjacency_list[4].push_back(neighbor(5, 9));
    // 5 = f
    adjacency_list[5].push_back(neighbor(0, 14));
    adjacency_list[5].push_back(neighbor(2, 2));
//    adjacency_list[5].push_back(neighbor(4, 9));
    printAdjacencyList(adjacency_list);

    std::vector <weight_t> min_distance;
    std::vector <vertex_t> previous;
    std::printf("init OK\n\n");

    const clock_t begin_time = clock();
    dijkstrap(start, adjacency_list, min_distance, previous);
    std::printf("distance from start node [%d] to end node [%d] is %.2f\n"
                "calculation time: %f sec\n",
                start, target, min_distance[target], float( clock () - begin_time ) /  CLOCKS_PER_SEC);

    std::list <vertex_t> path = getShortestPathToX(target, previous);
    std::cout << "Path : ";
    std::copy(path.begin(), path.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}
