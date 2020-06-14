// #include <fstream>
// #include <sstream>
// #include <string>
// #include <vector>
// #include <iostream>
// #include <iterator>
// #include <algorithm>
// #include <map>

// #include "data.h"


// /**
// splits string on given delimiter
// return string vector
// here: char from to weight
// */
// std::vector<std::string> split(const std::string& s, char delimiter) {
//    std::vector<std::string> tokens;
//    std::string token;
//    std::istringstream tokenStream(s);

//    while (std::getline(tokenStream, token, delimiter)) {
//       tokens.push_back(token);
//    }

//    return tokens;
// }

// void readGR(std::ifstream& infile, adjacency_list_t& list) {
//     std::string line;
//     int size;
//     std::map<int, std::vector<neighbor>> ans;
//     std::vector<std::string> elems;

// 	char type;
//     int source;
//     int target;
//     int weight;

//     do {
//     	std::istringstream iss(line);
//     	// std::printf("%s\n", line.c_str());

//     	if (line.rfind("p sp", 0) == 0) {
// 			elems = split(line, ' ');            
// 			std::cout << "reading graph with " << elems[2] << " nodes and " << elems[3] << " edges" << std::endl;
// 		}

// 		if(line.rfind("a", 0) == 0) {
// 			// std::cout << "DJANGO: " << line << std::endl;
// 			// std::vector<std::string> elems{ 
// 			//     std::istream_iterator<std::string>(iss), {}
// 			// };
//             elems = split(line, ' ');
// 			// std::cout << "connection: " << elems[0] << " " << elems[1] << elems[2] << elems[3] << std::endl;
// 			neighbor n = neighbor(elems[2], elems[3]);
// 			list[std::stoi(elems[1])].push_back(n);
// 		}
//      } while(std::getline(infile, line));

//     // printAdjacencyList(ans);
// }

// int main() {
// 	std::ifstream infile("../resources/sampleGraph.gr");
// 	adjacency_list_t adjacency_list;

// 	readGR(infile, adjacency_list);
//     printAdjacencyList(adjacency_list);

//     return 0;
// }
