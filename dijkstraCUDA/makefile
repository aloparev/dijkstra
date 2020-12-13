CPP=g++ #1
NVC=nvcc #2
FLG=-g

# afte<Down>r -I a custom omp.h is added
#FLG_DBG=-fsanitize=thread -I../../libgomp/build/x86_64-pc-linux-gnu/libgomp
#LOG=$(wildcard *.log)

SRC=dijkstrac2.cpp
DEP=../utils/data.cpp
EXE=dijkstrac

SRC2=$(wildcard *.cu)
EXE2=dicu

.PHONY: all
all: debug

.PHONY: build
build: $(SRC2)
	$(NVC) $(SRC) -o $(EXE) $(FLG)

# .PHONY: test
# test: $(SRC_TST)
# 	$(foreach T,$(EXE_TST), ./$(T) &&) true

.PHONY: debug
debug: $(SRC)
	$(CPP) $(SRC) $(DEP) -o $(EXE) $(FLG)

clean:
	rm -Rf $(EXE) $(LOG)
