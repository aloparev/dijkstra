/** 
 * @brief: Test program for Dijkstra CUDA
 * @author 557966
 * @date 31 V 2020
 *
 * g++ dijkstrac2.cpp ../utils/data.cpp -o dijkstrac2
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
#include <fstream>
#include "../utils/data.h"

const int max_weight_int = 2147483647; //2,147,483,647

/**
 * GPU (device) kernel function: given directed, weighted graph, compute shortest path 
 * \param source        source vertex as path start
 * \param matrix        graph representation
 * \param size          number of graph nodes
 * \param size2         number maximal edges per node
 * \param min_distance  contains distances for each node
 * \param previous      contains the predecessor for each node
 * \param visited       contains status whether a node already was visited
 * \param debug         boolean for extra output prints
 */
void dijkstra(const int& source, const std::vector<int>& matrix, const int& size, const int& size2, std::vector<int>& min_distance, std::vector<int>& previous, std::vector<int>& visited, const bool& debug) {
    int dist, u, ti, wi, v, w, distance_through_u;

    while (true) {
        dist = max_weight_int;
        u = -1;

        if(debug) printf("\tsearching for new node to process");

        for (int j=0; j < size; j++) {
            if(visited[j] == 0 && min_distance[j] < dist) {
                dist = min_distance[j];
                u = j;
            }
        }
        
        if(u == -1) {
            if(debug) { printf("...nothing found, exit\n"); }
            break; // exit loop, if no unvisited nodes  
        }
        if(debug) { printf("...found %d\n", u); }
        visited[u] = 1;

        for(int tid = 0; tid < size2/2; tid++) {
            ti = size2 * u + tid * 2; // target index
            wi = size2 * u + tid * 2 + 1; // weights index
            v = matrix[ti]; // neighbor fixation
            w = matrix[wi]; // weight
            if(debug) { printf("tid %d: t_index = %d/%d, t_value = %d/%d\n", tid, ti, wi, v, w); }

            if(v == -1) {
                if(debug) { printf("tid=%d: skip empty slot\n", tid); }
                continue;
            }

            if(v == u) {
                if(debug) { printf("\tcontinue FROM %d TO %d\n", u, v); }
                // printf("continue: u=%d ud=%d tid=%d\n", u, min_distance[u], tid);
                continue;
            }

            distance_through_u = w + min_distance[u];
            if(debug) { printf("tid=%d: w = %d, distance_through_u = %d, min_distance[%d] = %d\n", tid, w, distance_through_u, v, min_distance[v]); }

            if (distance_through_u < min_distance[v]) {
                min_distance[v] = distance_through_u;
                previous[v] = u;
                if(debug) { printf("\trelaxation FROM %d TO %d WITH %d + %d = %d\n", u, v, min_distance[u], w, min_distance[u] + w); }
            }

        // for(int v=0; v<size; v++) {
        //     printf("node=%d d=%.2f v=%s p=%d\n", v+1, min_distance[v], visited[v] ? "true" : "false", previous[v]+1);
        // }
        // std::cout << std::endl;
        }
    }   
}

/**
 * Unfolds shortest path from target to source backwards
 * \param vertex    target node
 * \param previous  contains the predecessor for each node
 * \return path     path from target to source
 */
std::list <int> getShortestPathCuda(int vertex, const std::vector <int>& previous) {
    std::list <int> path;
    
     do {
        path.push_front(vertex);
        vertex = previous[vertex];
     } while(vertex != -1);
    return path;
}

/**
 * Prints graph in form of sparse matrix
 * \param matrix    graph representation
 * \param size      number of graph nodes
 * \param size2     number maximal edges per node
 */
void printSparseMatrix(const std::vector<int>& matrix, const int size, const int size2) {
  printf("\nprinting flattened sparse matrix of size %d and max line length %d\n", size, size2);
  printf("structure: node -> { (edge_1, weight_1), ... (edge_size2, weight_size2) }\n");

  for(int row=0, r=0; row < size*size2; row += size2, r++) { //node
    printf("node %d:", r);

    for(int col=0; col < size2; col++) { //neighbors
      printf(" %2d ", matrix[row+col]);
    }
    printf("\n");
  }
  // std::cout << std::endl;
}

/**
 * Launcher
 * \param argc      program argument counter
 * \param argv      submitted arguments: 1=graph file path, 2=source node, 3=target node, 4=extra prints
 * \return status   program exit status
 */
int main(int argc, char** argv) {
    const clock_t begin_time = clock();
    
    FILE *fp;
    const char *input_file_name;
    char* line = NULL;
    size_t llen = 0;
    ssize_t read = 0;

    int size, edges, source, target, weight, status, start, end, max_edges, size2, ti, wi, debug_int;
    std::vector<int> matrix, min_distance, previous, visited;
    bool debug = true;

    printf("Dijkstra CUDA algorithm seq. tester\nargc=%d\n", argc);
    if (argc == 5) {
        input_file_name = argv[1];
        start = atoi(argv[2]);
        end = atoi(argv[3]);
        debug_int = atoi(argv[4]);
        if(debug_int == 0) { debug = false; }
        printf("init from args: ");
    }
    else {
        input_file_name = "resources/sampleGraph-1.gr";
        start = 0;
        end = 4;
        printf("no or bad args submitted, use default values: ");
    }
    printf("\n\tinput file = %s\n\tsource = %d\n\ttarget = %d\n\tverbose = %d", input_file_name, start, end, debug);

    fp = fopen(input_file_name,"r");
    if(!fp) printf("Error Openning Graph File\n");
    printf("\nReading Graph File");

    printf("\n");
    fscanf(fp, "p sp %d %d", &size, &edges);
    printf("(first line) size=%d edges=%d\n", size, edges);
    // matrix.resize(size*size, -1); //size2 = 1 159 330 980

    std::map<int, std::vector<neighbor> > adjacency_list;
    std::map<int, int> max_edges_map;

    while ((read = getline(&line, &llen, fp)) != -1) {
        if(debug) { printf("%3zu: %s", read, line); }
        if(line[0] == 'a' && sscanf(line, "a %d %d %d", &source, &target, &weight) == 3) {
                source--;
                target--;
                if(debug) { printf("\tarc from %d to %d weight %d | index %d\n", source, target, weight, source*size + target); }
                neighbor n = neighbor(target, weight);
                adjacency_list[source].push_back(n);
                max_edges_map[source]++;
        }
    }    

    max_edges = -1;
    for(int i=0; i<size; i++) {
        if(max_edges_map[i] > max_edges) {
            max_edges = max_edges_map[i];
        }
    }
    size2 = max_edges * 2;
    printf("\n");
    printf("filled temp map: max_edges = %d, size2 = %d,\n\tnew size = %d, square size = %d, diff = %d\n", max_edges, size2, size*size2, size*size, size*size - size*size2);

    matrix.resize(size * size2, -1);
    printf("\n");
    printf("start transfer from map to sparse matrix\n");
    // node
    for(int i=0; i<size; i++) {
        const std::vector <neighbor>& neighbors = adjacency_list.find(i)->second;
        int temp_size = neighbors.size();
        if(debug) { printf("node %d of size %d\n", i, temp_size); }

        // node edges
        for(int j=0; j < temp_size; j++) {
            target = neighbors[j].target;
            weight = neighbors[j].weight;
            ti = size2 * i + j * 2;
            wi = size2 * i + j * 2 + 1;
            if(debug) { printf("\ttarget = %d weight = %d | t_index = %d/%d\n", target, weight, ti, wi); }
            matrix[ti] = target;
            matrix[wi] = weight;
        }
    }

    if(fp) fclose(fp);    
    printf("Reading Graph File...OK\nempty temp map");
    adjacency_list.clear();
    printf("...OK\n");
    // printAdjacencyMatrix2(matrix, size);
    if(debug) { printSparseMatrix(matrix, size, size2); }

    printf("resize output vectors and set source to zero\n");
    min_distance.resize(size, max_weight_int);
    min_distance[start] = 0;
    previous.resize(size, -1);
    visited.resize(size, 0);
    
    // printf("main print vec of size %d\n", size);
    // for(int i=0; i<size; i++) printf("vec[%d] = %d\n", i, min_distance[i]);
    printf("init in %f sec OK\n\n", float( clock () - begin_time ) /  CLOCKS_PER_SEC);

    printf("start dijkstra\n");
    const clock_t begin_time2 = clock();
    dijkstra(start, matrix, size, size2, min_distance, previous, visited, debug);
    printf("distance from start node [%d] to end node [%d] is %2d\n"
                "calculation time: %f sec\n", start, end, min_distance[end], float( clock () - begin_time2 ) /  CLOCKS_PER_SEC);

    std::list<int> path = getShortestPathCuda(end, previous);
    size = path.size();
    printf("path of size %d:", size);
    // std::copy(path.begin(), path.end(), std::ostream_iterator<int>(std::cout, " "));
    for(int i=0; i < size; i++) {
        ti = path.front();
        printf(" %d/%d", ti, min_distance[ti]);
        path.pop_front();
    }
    std::cout << std::endl;
    printf("total run time is %f sec\n", float( clock () - begin_time ) /  CLOCKS_PER_SEC);

    return 0;
}