#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cstdio>
#include <math.h>

// #include "data.h"


// /**
// splits string on given delimiter
// return string vector
// here: char from to weight
// */
std::vector<std::string> split(const std::string& s, char delimiter) {
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);

   while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
   }

   return tokens;
}

void printAdjacencyMatrix(const std::vector<std::vector<int>>& matrix) {
  int size = static_cast<int>(matrix.size());
  // std::cout << "size=" << size << std::endl;

  for(int row=0; row < size; row++) { //node
    std::cout << std::endl;

    for(int col=0; col<size; col++) { //neeighbors
      // std::cout << matrix[row][col] << " ";
      printf("%2d ", matrix[row][col]);
    }
  }
  std::cout << std::endl;
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

int main() {
	std::ifstream infile("../resources/sampleGraph.gr");
  int size;
	std::vector<int> matrix;

  readGR4(infile, matrix, size);
  printAdjacencyMatrix2(matrix, size);

  return 0;
}
