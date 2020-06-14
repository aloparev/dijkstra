CPP=g++
NVC=nvcc
FLG=-g -I../dijkstraOMP/

# after -I a custom omp.h is added
FLG_DBG=-fsanitize=thread -I../../libgomp/build/x86_64-pc-linux-gnu/libgomp
LOG=$(wildcard *.log)

SRC=$(wildcard *.cu))
EXE=dicu

.PHONY: all
all: build test

.PHONY: build
build: $(SRC)
	$(NVC) $(SRC) -o $(EXE) $(FLG)

# .PHONY: test
# test: $(SRC_TST)
# 	$(foreach T,$(EXE_TST), ./$(T) &&) true

# .PHONY: debug
# debug: $(SRC)
# 	$(CPP) $(SRC) -o $(EXE) $(FLG) $(FLG_DBG)

clean:
	rm -Rf $(EXE) $(LOG)