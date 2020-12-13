/** 
 * @brief: helper unilities header and custom data structures for C++ Dijkstra implementations
 * @author 557966
 * @date 31 V 2020
 */

#include <limits> // for numeric_limits
#include <vector>
#include <map>

//! self defined types to quickly change type if need be
typedef int vertex_t; // vertex=node
typedef double weight_t;

/** 
Structure to hold an outgoing connection.
Based on self-defined types.
Includes constructor to init from integers and from string.
*/
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
typedef std::map<int, std::vector<neighbor>> adjacency_list_t;

void printAdjacencyList(const std::map<int, std::vector<neighbor>>&);
std::vector<std::string> split(const std::string&, char);
void readGR(std::ifstream&, adjacency_list_t&);
std::list <vertex_t> getShortestPathToX(vertex_t, const std::vector<vertex_t>&);

void printAdjacencyMatrix(const std::vector<std::vector<int>>& matrix);
void readGR2(std::ifstream& infile, std::vector<std::vector<int>>& matrix);
void printAdjacencyMatrix2(const std::vector<int>& matrix, const int size);

//! function X
void readGR4(std::ifstream& infile, std::vector<int>& matrix, int &size);