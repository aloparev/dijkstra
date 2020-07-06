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

typedef int vertex_t; // vertex=node
typedef double weight_t;

struct neighbor {
    vertex_t target;
    weight_t weight;

    neighbor(vertex_t arg_target, weight_t arg_weight)
            : target(arg_target), weight(arg_weight) {}

    neighbor(std::string arg_target, std::string arg_weight)
            : target(std::stoi(arg_target)), weight(std::stoi(arg_weight)) {}
};

// typedef std::vector <std::vector<neighbor>> adjacency_list_t;
const weight_t max_weight = std::numeric_limits<double>::infinity();

void printAdjacencyList(const std::map<int, std::vector<neighbor>>& adjacency_list) {
    int size = adjacency_list.size();
    std::printf("printing graph of size %d\n", size);

    for(auto const& p : adjacency_list) {
        std::printf("node[%d]: ", p.first);

        for(auto const& v : p.second) {
            std::printf("(%d, %.2f) ", v.target, v.weight);
        }
        std::printf("\n");
    }
        // std::printf("\n");
}

/**
splits string on given delimiter
return string vector
here: char from to weight
*/
std::vector<std::string> split(const std::string& s, char delimiter) {
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);

   while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
   }

   return tokens;
}

void readGR(std::ifstream& infile, adjacency_list_t& list) {
    std::string line;
    int size;
    std::map<int, std::vector<neighbor>> ans;
    std::vector<std::string> elems;

    char type;
    int source;
    int target;
    int weight;

    do {
        std::istringstream iss(line);
        // std::printf("%s\n", line.c_str());

        if (line.rfind("p sp", 0) == 0) {
            elems = split(line, ' ');            
            std::cout << "reading graph with " << elems[2] << " nodes and " << elems[3] << " edges" << std::endl;
        }

        if(line.rfind("a", 0) == 0) {
            // std::cout << "DJANGO: " << line << std::endl;
            // std::vector<std::string> elems{ 
            //     std::istream_iterator<std::string>(iss), {}
            // };
            elems = split(line, ' ');
            // std::cout << "connection: " << elems[0] << " " << elems[1] << elems[2] << elems[3] << std::endl;
            neighbor n = neighbor(elems[2], elems[3]);
            list[std::stoi(elems[1])].push_back(n);
        }
     } while(std::getline(infile, line));

    // printAdjacencyList(ans);
}

void printAdjacencyMatrix(const std::vector<std::vector<int>>& matrix) {
  int size = static_cast<int>(matrix.size());
  // std::cout << "size=" << size << std::endl;

  for(int row=0; row<size; row++) { //node
    std::cout << std::endl;

    for(int col=0; col<size; col++) { //neeighbors
      // std::cout << matrix[row][col] << " ";
      printf("%2d ", matrix[row][col]);
    }
  }
  std::cout << std::endl;
}

void printNbrs(const std::vector<int>& nbrs) {
  int size = static_cast<int>(nbrs.size());
  // std::cout << "size=" << size << std::endl;

  std::cout << std::endl;
  for(int row=0; row<size; row++) { //node
      printf("node %2d neighbors: %2d\n", row, nbrs[row]);
    }
}

void readGR2(std::ifstream& infile, std::vector<std::vector<int>>& matrix) {
    std::string line;
    int size = -1;
    std::vector<std::string> elems;
    int source, target, weight;

    do {
        std::istringstream iss(line);
        // std::printf("%s\n", line.c_str());

        if (line.rfind("p sp", 0) == 0) {
            elems = split(line, ' ');            
            size = std::stoi(elems[2]);
            std::cout << "red graph with " << size << " nodes and " << elems[3] << " edges" << std::endl;

            matrix.resize(size, std::vector<int> (size, -1));
            std::cout << "init matrix " << size << "x" << size << std::endl;
        }

        if(line.rfind("a", 0) == 0) {
            elems = split(line, ' ');
            source = std::stoi(elems[1]) - 1;
            target = std::stoi(elems[2]) - 1;
            weight = std::stoi(elems[3]);
            std::cout << "connection: " << elems[0] << " " << source << " " << target << " " << weight << std::endl;
            // std::cout << "connection: " << elems[0] << " " << elems[1] << " " << elems[2] << " " << elems[3] << std::endl;
            matrix[source][target] = weight;
        }
     } while(std::getline(infile, line));

    // printAdjacencyList(ans);
     // return matrix;
}

void printAdjacencyMatrix2(const std::vector<int>& matrix, const int size) {
  printf("\nprinting flattened vector of size %d and line lenght %d\n", size*size, size);

  for(int row=0; row < size*size; row += size) { //node
    std::cout << std::endl;

    for(int col=0; col < size; col++) { //neeighbors
      // std::cout << matrix[row][col] << " ";
      printf("%2d ", matrix[row+col]);
    }
  }
  std::cout << std::endl;
}

void readGR4(std::ifstream& infile, std::vector<int>& matrix, int &size) {
    std::string line;
    std::vector<std::string> elems;
    int source, target, weight;
    int i=1;

    do {
        std::istringstream iss(line);
        // std::printf("%s\n", line.c_str());

        if (line.rfind("p sp", 0) == 0) {
            elems = split(line, ' ');            
            size = std::stoi(elems[2]);
            std::cout << "red graph with " << size << " nodes and " << elems[3] << " edges" << std::endl;

            matrix.resize(size*size, -1);
            std::cout << "init matrix " << size << "x" << size << std::endl;
            std::cout << std::endl;
        }

        if(line.rfind("a", 0) == 0) {
            elems = split(line, ' ');
            source = std::stoi(elems[1]) - 1;
            target = std::stoi(elems[2]) - 1;
            weight = std::stoi(elems[3]);
            std::cout << "arc " << i << " from " << source << " to " << target << " weight " << weight << std::endl;
            i++;
            // std::cout << "connection: " << elems[0] << " " << elems[1] << " " << elems[2] << " " << elems[3] << std::endl;
            matrix[source*size + target] = weight;
        }
     } while(std::getline(infile, line));

    // printAdjacencyList(ans);
     // return matrix;
}

/*
nvcc ./app.cu –o app
./app
https://stackoverflow.com/questions/18963293/cuda-atomics-change-flag/18968893#18968893
*/

__global__ void checkNeighbors(const int* matrix, const int size, const int u, int* min_distance_dev, int* previous_dev) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid < size) {
        if(tid == u || matrix[u*size + tid] == -1) return;

        int w = matrix[u*size + tid];
            weight_t distance_through_u = w + min_distance_dev[u];

            if (distance_through_u < min_distance_dev[tid]) {
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
