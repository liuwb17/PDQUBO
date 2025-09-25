CC = gcc
CXX = g++
CXXFLAGS = -std=c++17 -ltbb -Wall -Wextra -Wpedantic
INCLUDES = -I../include
BOOST = -lboost_program_options
THREAD = -lpthread

ifdef STRICT
CXXFLAGS += -Wconversion -Wsign-conversion -Wtype-limits -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual
endif

ifdef DEBUG
CXXFLAGS += -DDEBUG=$(DEBUG) -g
else
CXXFLAGS += -O3
endif

ifdef MAXDEG
CXXFLAGS += -DMAXDEG=$(MAXDEG)
else
CXXFLAGS += -DMAXDEG=2
endif


LDFLAGS_ABS2 = -L../abs2/lib -labs2 -ldl
LDFLAGS_GRB = -lgurobi_c++ -lgurobi110


TARGET_EASY = partition_easy simple_shift_scheduling_easy shift_scheduling_easy bin_packing_easy nqueen_easy ilp_easy tsp_easy simple_factorization_easy graph_color_easy
TARGET_EXHAUSTIVE = partition_exhaustive knapsack_exhaustive ilp_exhaustive change_making_exhaustive
TARGET_ABS2 = ilp_abs2 tsp_abs2 simple_factorization_abs2 graph_color_abs2
TARGET_GRB = ilp_grb tsp_grb simple_factorization_grb graph_color_grb nqueen_grb
TARGET = $(TARGET_EASY) $(TARGET_EXHAUSTIVE) $(TARGET_ABS2) $(TARGET_GRB) factorization

all: $(TARGET)

easy: $(TARGET_EASY)
exhaustive: $(TARGET_EXHAUSTIVE)
abs2: $(TARGET_ABS2)
grb: $(TARGET_GRB)

clean:
	rm -f $(TARGET) xxhash.o


%_easy: %_easy.cpp xxhash.o
	$(CXX) $< xxhash.o -o $@ $(CXXFLAGS) $(INCLUDES)

%_exhaustive: %_exhaustive.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDES)

%_abs2: %_abs2.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS_ABS2)

%_grb: %_grb.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS_GRB)


partition_easy: partition_easy.cpp xxhash.o
	$(CXX) $< xxhash.o -o $@ $(CXXFLAGS) $(INCLUDES) -DENERGY_TYPE=cpp_int -DCOEFF_TYPE=cpp_int $(BOOST)

partition_exhaustive: partition_exhaustive.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDES) -DENERGY_TYPE=cpp_int -DCOEFF_TYPE=cpp_int $(BOOST)

nqueen_easy: nqueen_easy.cpp qbpp_nqueen.hpp xxhash.o
	$(CXX) $< xxhash.o -o $@ $(CXXFLAGS) $(INCLUDES) -DCOEFF_TYPE=int8_t -DENERGY_TYPE=int16_t $(BOOST)

nqueen_grb: nqueen_grb.cpp qbpp_nqueen.hpp
	$(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS_GRB) -DCOEFF_TYPE=int8_t -DENERGY_TYPE=int16_t $(BOOST)

graph_color_easy: graph_color_easy.cpp xxhash.o
	$(CXX) $< xxhash.o -o $@ $(CXXFLAGS) $(INCLUDES) $(BOOST)

graph_color_abs2: graph_color_abs2.cpp xxhash.o
	$(CXX) $< xxhash.o -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS_ABS2) $(BOOST)

graph_color_grb: graph_color_grb.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS_GRB) $(BOOST)

tsp_easy: tsp_easy.cpp qbpp_tsp.hpp xxhash.o
	$(CXX) $< xxhash.o -o $@ $(CXXFLAGS) $(INCLUDES) $(BOOST)

tsp_abs2: tsp_abs2.cpp qbpp_tsp.hpp
	$(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS_ABS2) $(BOOST)

tsp_grb: tsp_grb.cpp qbpp_tsp.hpp
	$(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS_GRB) $(BOOST)

simple_factorization_easy: simple_factorization_easy.cpp xxhash.o
	$(CXX) $< xxhash.o -o $@ $(CXXFLAGS) $(INCLUDES)

factorization: factorization.cpp qbpp_multiplier.hpp xxhash.o
	$(CXX) $< xxhash.o -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS_ABS2) $(LDFLAGS_GRB) $(THREAD) $(BOOST)

xxhash.o: xxhash.c xxhash.h
	$(CC) $< -c -o $@ -O3
