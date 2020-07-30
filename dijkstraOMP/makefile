# g++ -fopenmp dijkstra.cpp -o dipa -g -fsanitize=thread


# run: 
#     OMP_NUM_THREADS=4 ./dipa
#     valgrind --time-stamp=yes --tool=helgrind --log-file=helgrind.log ./dipa
#     LD_LIBRARY_PATH=../../libgomp/build/x86_64-pc-linux-gnu/libgomp/.libs ./dipa
#		ldd to test libgomp


CPP=g++
FLG=-fopenmp -g

# after -I a custom omp.h is added
FLG_DBG=-fsanitize=thread -I../../libgomp/build/x86_64-pc-linux-gnu/libgomp
LOG=$(wildcard *.log)

SRC=$(filter-out %_test.cpp,$(wildcard *.cpp))
DEP=../utils/data.cpp
EXE=dipa

SRC_TST=$(wildcard *_test.cpp)
EXE_TST=$(patsubst %.cpp,%,$(TST_CRC))


# target: dependency
# 	compile dependency cmd

# $(EXE): $(SRC)
# 	$(CPP) $(SRC) -o $(EXE) $(FLG)


.PHONY: all
all: build test

.PHONY: build
build: $(SRC)
	$(CPP) $(SRC) $(DEP) -o $(EXE) $(FLG)

.PHONY: test
test: $(SRC_TST)
	$(foreach T,$(EXE_TST), ./$(T) &&) true

.PHONY: debug
debug: $(SRC)
	$(CPP) $(SRC) $(DEP) -o $(EXE) $(FLG) $(FLG_DBG)

clean:
	rm -Rf $(EXE) $(LOG)