#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>
#include <list>

#include "data.h"

void printAdjacencyList(const adjacency_list_t &adjacency_list) {
    int size = adjacency_list.size();
    printf("printing graph of size %d\n", size);

    for(auto const& p : adjacency_list) {
        std::printf("node[%d]: ", p.first);

        for(auto const& v : p.second) {
            std::printf("(%d, %.2f) ", v.target, v.weight);
        }
        std::printf("\n");
    }

        // std::printf("\n");
}

std::list <vertex_t> getShortestPathToX(vertex_t vertex, const std::vector <vertex_t> &previous) {
    std::list <vertex_t> path;
    
     do {
        path.push_front(vertex);
        vertex = previous[vertex];
     } while(vertex != -1);
    return path;
}

void readGR(std::ifstream& infile, adjacency_list_t& list) {
    std::string line;
    int source, target, weight, size;
    std::map<int, std::vector<neighbor>> ans;
    std::vector<std::string> elems;
    char type;

    do {
        std::istringstream iss(line);
        // std::printf("%s\n", line.c_str());

        if (line.rfind("p sp", 0) == 0) {
            elems = split(line, ' ');            
            std::cout << "reading graph with " << elems[2] << " nodes and " << elems[3] << " edges" << std::endl;
        }

        if(line.rfind("a", 0) == 0) {
            elems = split(line, ' ');
            source = std::stoi(elems[1]) - 1;
            target = std::stoi(elems[2]) - 1;
            weight = std::stoi(elems[3]);

            neighbor n = neighbor(target, weight);
            list[source].push_back(n);
        }
     } while(std::getline(infile, line));

    // printAdjacencyList(ans);
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
