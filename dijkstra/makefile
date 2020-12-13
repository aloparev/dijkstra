# g++ dijkstra.cpp -o dijkstra


CPPFLAGS=-g
CPP=g++
SRC=$(filter-out %_test.cpp,$(wildcard *.cpp))
DEP=../utils/data.cpp
EXE=dijkstra

# dijkstra: dijkstra.cpp
# 	$(CPP) dijkstra.cpp -o dijkstra

.PHONY: all
all: build

.PHONY: build
build: $(SRC)
	$(CPP) $(SRC) $(DEP) -o $(EXE) $(FLG)

clean:
	rm -Rf $(EXE)